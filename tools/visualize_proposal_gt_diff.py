#!/usr/bin/env python3
"""Compare cached proposal unions with VOC foreground GT (CPU, read-only inputs)."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"tp": (35, 190, 80), "fp": (240, 55, 55),
          "fn": (45, 110, 245), "ignore": (145, 145, 145)}
FIELDS = ["image_id", "proposal_count", "tp", "fp", "fn", "valid_pixels",
          "ignored_pixels", "precision", "recall", "iou", "status", "error"]


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def ratios(tp, fp, fn):
    def divide(a, b):
        return a / b if b else None
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                iou=divide(tp, tp + fp + fn))


def load_masks(path, shape):
    with np.load(path, allow_pickle=False) as data:
        masks, scores = data["masks"], data["scores"]
        if masks.ndim != 3 or scores.shape != (len(masks),):
            raise ValueError("Expected masks [N,H,W] and scores [N]")
        if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite real numbers")
        if masks.dtype.kind not in "bifu" or not np.all((masks == 0) | (masks == 1)):
            raise ValueError("Masks must be binary")
        if len(masks) and masks.shape[1:] != tuple(shape):
            raise ValueError(f"Mask dimensions {masks.shape[1:]} != image {shape}")
        return masks.astype(bool) if len(masks) else np.zeros((0, *shape), dtype=bool)


def load_union(path, shape):
    masks = load_masks(path, shape)
    return masks.any(axis=0), len(masks)


def compare(union, gt):
    if gt.ndim != 2 or gt.shape != union.shape:
        raise ValueError("GT must be a 2D label image matching original dimensions")
    if not np.all(((gt >= 0) & (gt <= 20)) | (gt == 255)):
        raise ValueError("GT labels must be 0..20 or 255")
    valid = gt != 255
    foreground = (gt > 0) & valid
    regions = dict(tp=union & foreground, fp=union & ~foreground & valid,
                   fn=~union & foreground, ignore=~valid)
    counts = {key: int(regions[key].sum()) for key in ("tp", "fp", "fn")}
    counts.update(valid_pixels=int(valid.sum()), ignored_pixels=int((~valid).sum()))
    counts.update(ratios(counts["tp"], counts["fp"], counts["fn"]))
    return regions, counts


def overlay(rgb, regions):
    result = rgb.copy()
    for key, mask in regions.items():
        result[mask] = np.rint(0.5 * rgb[mask] + 0.5 * np.array(COLORS[key])).astype(np.uint8)
    return Image.fromarray(result)


def mask_edge(mask):
    """One-pixel inner outline, including holes and image-border edges."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = (mask & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def render(rgb, masks, regions, image_id, metrics):
    # Keep all proposals visible, including those with zero GT intersection
    # and those entirely in ignored regions. GT only controls the fill colors.
    pixels = np.array(overlay(rgb, regions))
    labels = []
    for index, mask in enumerate(masks, 1):
        edge = mask_edge(mask)
        pixels[edge] = (255, 220, 0)
        ys, xs = np.nonzero(edge)
        if len(ys):
            labels.append((int(xs[0]), int(ys[0]), f"#{index}"))
    panel = Image.fromarray(pixels)
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    for x, y, text in labels:
        box = draw.textbbox((0, 0), text)
        x = max(0, min(x, width - (box[2] - box[0]) - 2))
        y = max(0, min(y, height - (box[3] - box[1]) - 4))
        draw.text((x, y), text, fill=(255, 220, 0), stroke_width=1, stroke_fill="black")
    header = 126
    canvas = Image.new("RGB", (max(width, 480), height + header), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{image_id} | proposals={len(masks)}", fill="black")
    iou = "undefined" if metrics["iou"] is None else f"{metrics['iou']:.4f}"
    draw.text((10, 26), f"IoU={iou}  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}", fill="black")
    for index, (key, label) in enumerate([
            ("tp", "Correct (TP)"), ("fp", "Over (FP)"),
            ("fn", "Under (FN)"), ("ignore", "Ignored (255)")]):
        x, y = 10 + (index % 2) * 230, 48 + (index // 2) * 23
        draw.rectangle((x, y, x + 16, y + 16), fill=COLORS[key])
        draw.text((x + 23, y + 2), label, fill="black")
    draw.text((10, 100), "Yellow outlines / #ID: ALL cached proposals (1-based)", fill="black")
    canvas.paste(panel, ((canvas.width - width) // 2, header))
    return canvas


def create_output(explicit):
    if explicit is not None:
        explicit.mkdir(parents=True, exist_ok=False)
        return explicit.resolve()
    base = ROOT / "experiments" / datetime.now().strftime("proposal_gt_diff_%Y%m%d_%H%M%S")
    for suffix in range(10000):
        path = base if suffix == 0 else base.with_name(f"{base.name}_{suffix}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a new experiment directory")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/VOC2012/JPEGImages")
    parser.add_argument("--gt-dir", type=Path, default=ROOT / "data/VOC2012/SegmentationClassAug")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all NPZ files")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.image_id and (args.image_id in (".", "..") or
                          any(c in args.image_id for c in "/\\:")):
        parser.error("--image-id must be a plain filename stem")
    for name in ("proposals_dir", "image_dir", "gt_dir"):
        path = getattr(args, name).resolve()
        if not path.is_dir():
            parser.error(f"Directory not found: {path}")
        setattr(args, name, path)
    return args


def main(argv=None):
    args = parse_args(argv)
    ids = [args.image_id] if args.image_id else sorted(p.stem for p in args.proposals_dir.glob("*.npz"))
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("No proposal NPZ files found")
    images = {}
    for path in sorted(args.image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            images.setdefault(path.stem, []).append(path)
    output = create_output(args.output_dir)
    (output / "overlays").mkdir()
    run = dict(status="running", parameters={k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
               output_dir=str(output), selected_ids=ids, files=[], succeeded=0, failed=0,
               script=identity(Path(__file__)),
               protocol=dict(gt_foreground="1..20", ignore=255, alpha=0.5,
                             visualization="single overlay; all proposal outlines and 1-based IDs",
                             filtering="none", orientation="stored pixels, no EXIF transform",
                             undefined_metrics=None, aggregate="pooled successful-image pixel counts"))
    totals = dict(tp=0, fp=0, fn=0, valid_pixels=0, ignored_pixels=0)

    def save_run():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(output / "run.json")

    save_run()
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for number, image_id in enumerate(ids, 1):
            row = dict(image_id=image_id)
            record = dict(image_id=image_id, inputs={})
            try:
                candidates = images.get(image_id, [])
                paths = dict(proposals=args.proposals_dir / f"{image_id}.npz",
                             gt=args.gt_dir / f"{image_id}.png")
                if len(candidates) != 1:
                    raise ValueError(f"Expected one original image for {image_id}, found {len(candidates)}")
                paths["image"] = candidates[0]
                for key, path in paths.items():
                    record["inputs"][key] = identity(path)
                with Image.open(paths["image"]) as source:
                    rgb = np.array(source.convert("RGB"))
                with Image.open(paths["gt"]) as source:
                    gt = np.array(source)
                masks = load_masks(paths["proposals"], rgb.shape[:2])
                union, count = masks.any(axis=0), len(masks)
                regions, metrics = compare(union, gt)
                render(rgb, masks, regions, image_id, metrics).save(
                    output / "overlays" / f"{image_id}.png")
                row.update(proposal_count=count, **metrics, status="ok")
                for key in totals:
                    totals[key] += metrics[key]
                run["succeeded"] += 1
            except (OSError, ValueError, KeyError, EOFError, BadZipFile, TypeError) as exc:
                row.update(status="error", error=str(exc))
                run["failed"] += 1
                print(f"[ERROR] {image_id}: {exc}", flush=True)
            record.update(status=row["status"], error=row.get("error"))
            run["files"].append(record)
            writer.writerow(row)
            stream.flush()
            print(f"[{number}/{len(ids)}] {image_id}: {row['status']}", flush=True)
    run.update(status="incomplete" if run["failed"] else "complete",
               aggregate={**totals, **ratios(totals["tp"], totals["fp"], totals["fn"])})
    save_run()
    print(f"Output: {output} | success={run['succeeded']} failed={run['failed']}")
    return 1 if run["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
