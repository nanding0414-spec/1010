#!/usr/bin/env python3
"""Visualize cached proposal masks without loading any model.

Colors identify proposals, not semantic classes. Overlap fill uses the highest
proposal score; outlines include all selected masks. Individual views preserve
the visibility of nested/overlapping proposals.
"""

import argparse
import colorsys
import csv
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_cached_proposals(path, image_shape):
    with Path(path).open("rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            masks = data["masks"]
            scores = data["scores"]
    if masks.ndim != 3 or scores.shape != (len(masks),):
        raise ValueError("Expected masks [N,H,W] and scores [N]")
    # The existing writer stores empty proposals with shape (0, 1, 1).
    if len(masks) and masks.shape[1:] != tuple(image_shape):
        raise ValueError(f"Mask size {masks.shape[1:]} != image size {image_shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Proposal scores contain NaN or infinity")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Expected binary cached masks")
    return masks.astype(bool), scores.astype(np.float32)


def proposal_color(index):
    rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.75, 1.0)
    return tuple(int(round(c * 255)) for c in rgb)


def select_proposals(masks, scores, min_score=None, top_k=0):
    indices = np.flatnonzero(masks.any(axis=(1, 2)))
    if min_score is not None:
        indices = indices[scores[indices] >= min_score]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return indices[:top_k] if top_k else indices


def render_overlay(rgb, masks, scores, indices, alpha=0.35, line_width=2,
                   draw_labels=False):
    result = rgb.copy()
    occupied = np.zeros(rgb.shape[:2], dtype=bool)
    # Blend each pixel once so overlapping masks do not obscure the photograph.
    for index in indices:
        visible = masks[index] & ~occupied
        color = np.asarray(proposal_color(int(index)), dtype=np.float32)
        result[visible] = np.rint(
            rgb[visible].astype(np.float32) * (1 - alpha) + color * alpha
        ).astype(np.uint8)
        occupied |= masks[index]

    for index in reversed(indices):
        binary = masks[index].astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        color = proposal_color(int(index))
        cv2.drawContours(result, contours, -1, color, line_width, cv2.LINE_AA)
    if draw_labels:
        for index in reversed(indices):
            binary = masks[index].astype(np.uint8)
            if not binary.any():
                continue
            # Place the ID in the mask interior instead of a possibly external centroid.
            padded = np.pad(binary, 1)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            label = f"#{index + 1} {scores[index]:.3f}"
            (width, height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            x = max(0, min(int(x), rgb.shape[1] - width - 2))
            y = max(height + 1, min(int(y), rgb.shape[0] - baseline - 1))
            cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return result, occupied


def image_index(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    index = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in index:
                raise ValueError(f"Ambiguous image ID: {path.stem}")
            index[path.stem] = path
    return index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/VOC2012/JPEGImages"))
    parser.add_argument("--proposals-dir", type=Path, required=True,
                        help="Exact directory containing cached .npz files")
    parser.add_argument("--output-dir", type=Path, default=Path("proposal_visualizations"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--split", type=Path, help="Image IDs, one per line (e.g. VOC val.txt)")
    selection.add_argument("--image-id", help="Visualize one image ID, without extension")
    parser.add_argument("--limit", type=int, default=0, help="First N images; 0 means all")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Optional visualization-only filter; default keeps all scores")
    parser.add_argument("--top-k", type=int, default=0, help="0 keeps all proposals")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--draw-labels", action="store_true", help="Show proposal ID and SAM3 score")
    parser.add_argument("--save-individual", action="store_true",
                        help="Also save one overlay per selected proposal; may use substantial disk space")
    args = parser.parse_args(argv)
    if not 0 <= args.alpha <= 1 or args.line_width < 1:
        parser.error("alpha must be in [0,1] and line-width must be >= 1")
    if args.limit < 0 or args.top_k < 0:
        parser.error("limit and top-k must be >= 0")
    if args.min_score is not None and not np.isfinite(args.min_score):
        parser.error("min-score must be finite")
    for directory in (args.image_dir, args.proposals_dir):
        if not directory.is_dir():
            parser.error(f"Directory not found: {directory}")
    return args


def main(argv=None):
    args = parse_args(argv)
    images = image_index(args.image_dir)
    if args.image_id:
        ids = [args.image_id]
    elif args.split:
        ids = [line.strip() for line in args.split.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    else:
        ids = [path.stem for path in sorted(args.proposals_dir.glob("*.npz"))]
    ids = list(dict.fromkeys(ids))
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No proposal image IDs found")
    if any(Path(i).name != i or "/" in i or "\\" in i or i in (".", "..") for i in ids):
        raise ValueError("Expected plain image IDs without directories")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    comparison_dir = args.output_dir / "comparisons"
    overlay_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    errors = 0
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "image_id", "cached_count", "shown_count", "coverage_ratio", "status", "error"
        ])
        writer.writeheader()
        for number, image_id in enumerate(ids, 1):
            try:
                if image_id not in images:
                    raise FileNotFoundError(f"Original image not found: {image_id}")
                # Match cv2.imread's EXIF orientation used when generating proposals.
                with Image.open(images[image_id]) as source:
                    rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
                masks, scores = load_cached_proposals(
                    args.proposals_dir / f"{image_id}.npz", rgb.shape[:2]
                )
                indices = select_proposals(masks, scores, args.min_score, args.top_k)
                overlay, occupied = render_overlay(
                    rgb, masks, scores, indices, args.alpha, args.line_width, args.draw_labels
                )
                Image.fromarray(overlay).save(overlay_dir / f"{image_id}.jpg", quality=95)
                comparison = np.concatenate((rgb, overlay), axis=1)
                Image.fromarray(comparison).save(comparison_dir / f"{image_id}.jpg", quality=95)
                if args.save_individual:
                    single_dir = args.output_dir / "individual" / image_id
                    single_dir.mkdir(parents=True, exist_ok=True)
                    for index in indices:
                        single, _ = render_overlay(
                            rgb, masks, scores, [index], args.alpha, args.line_width, args.draw_labels
                        )
                        filename = f"{index + 1:04d}_score_{scores[index]:.4f}.jpg"
                        Image.fromarray(single).save(single_dir / filename, quality=95)
                writer.writerow(dict(image_id=image_id, cached_count=len(masks),
                                     shown_count=len(indices), coverage_ratio=float(occupied.mean()),
                                     status="ok", error=""))
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                errors += 1
                writer.writerow(dict(image_id=image_id, status="error", error=str(exc)))
                print(f"[WARN] {image_id}: {exc}", flush=True)
            if number % 25 == 0 or number == len(ids):
                stream.flush()
                print(f"[Visualize] {number}/{len(ids)}, errors={errors}", flush=True)
    print(f"[Done] Output: {args.output_dir.resolve()}")
    print("Colors are proposal IDs, not classes. Coverage is not an accuracy metric.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
