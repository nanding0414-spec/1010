#!/usr/bin/env python3
"""Evaluate the quality of SAM3-generated pseudo labels against GT.

Metrics per-image and aggregated:
  - mIoU (with / without background)
  - per-class IoU
  - pixel accuracy (overall / foreground)
  - foreground precision / recall / F1
  - statistics: empty-mask ratio, fg-pixel ratio distribution, class count
  - image-level label consistency (pseudo classes vs allowed labels)
  - worst-N samples ranked by per-image mIoU
  - visualization: original + pseudo + GT overlay for worst-N

Usage:
    python tools/eval_pseudo_mask_quality.py --dataset voc
    python tools/eval_pseudo_mask_quality.py --dataset voc --worst 30 --vis
    python tools/eval_pseudo_mask_quality.py --dataset voc --split train
    python tools/eval_pseudo_mask_quality.py --dataset voc --pseudo-dir data/VOC2012/pseudo/sam3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset layouts (mirror generate_pseudo_masks_sam3.py)
# ---------------------------------------------------------------------------
VOC_ROOT = Path("data/VOC2012")
COCO_ROOT = Path("data/COCO2014")
ADE_ROOT = Path("data/ADEChallengeData2016")
CITY_ROOT = Path("data/CityScapes")

PSEUDO_SUBDIR = "sam3"


def get_dataset_paths(dataset_type, split, pseudo_dir=None):
    """Return (dataset_root, pseudo_dir, gt_dir, imagelevel_json, img_dir, class_file)."""
    dataset_type = dataset_type.lower()
    if dataset_type == "voc":
        root = VOC_ROOT
        pseudo = Path(pseudo_dir) if pseudo_dir else root / "pseudo" / PSEUDO_SUBDIR
        gt_dir = root / "SegmentationClassAug"
        img_dir = root / "JPEGImages"
        imglevel_json = root / "ImageSets" / "ImageLevel" / f"{split}_imagelevel.json"
        cls_file = "configs/cls_voc21.txt"
        has_ignore = True  # GT uses 255 for ignore
    elif dataset_type == "coco":
        root = COCO_ROOT
        pseudo = Path(pseudo_dir) if pseudo_dir else root / "pseudo" / PSEUDO_SUBDIR / split
        gt_dir = root / "SegmentationClass" / split
        img_dir = root / "images" / split
        split_name = "train" if "train" in split else "val"
        imglevel_json = root / "annotations" / f"{split_name}_imagelevel.json"
        cls_file = "configs/cls_coco_object.txt"
        has_ignore = True
    elif dataset_type == "ade20k":
        root = ADE_ROOT
        split_dir = "training" if split in ("train", "training") else "validation"
        pseudo = Path(pseudo_dir) if pseudo_dir else root / "pseudo" / PSEUDO_SUBDIR / split_dir
        gt_dir = root / "annotations" / split_dir
        img_dir = root / "images" / split_dir
        split_name = "train" if split_dir == "training" else "val"
        imglevel_json = root / "ImageSets" / "ImageLevel" / f"{split_name}_imagelevel.json"
        cls_file = "configs/cls_ade20k.txt"
        has_ignore = False
    elif dataset_type == "cityscapes":
        root = CITY_ROOT
        split_dir = "train" if split == "train" else "val"
        pseudo = Path(pseudo_dir) if pseudo_dir else root / "pseudo" / PSEUDO_SUBDIR / split_dir
        gt_dir = root / "gtFine" / split_dir  # need city subdir walk
        img_dir = root / "leftImg8bit" / split_dir
        imglevel_json = root / "ImageSets" / "ImageLevel" / f"{split_dir}_imagelevel.json"
        cls_file = "configs/cls_city_scapes.txt"
        has_ignore = True
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")
    return root, pseudo, gt_dir, img_dir, imglevel_json, cls_file, has_ignore


def load_class_names(cls_file, dataset_type):
    """Load foreground class names (skip background line for voc/coco)."""
    lines = [l.strip() for l in Path(cls_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if dataset_type in ("voc", "coco"):
        lines = lines[1:]  # skip background line
    return [line.split(";")[0].strip() for line in lines]


def load_image_list(imglevel_json, dataset_type):
    """Load (img_id, img_path_stem, allowed_labels) from image-level json."""
    if not Path(imglevel_json).is_file():
        raise FileNotFoundError(f"Image-level label json not found: {imglevel_json}")
    with open(imglevel_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for info in data["images"]:
        img_id = str(info.get("img_id", ""))
        file_name = info["file_name"]
        stem = Path(file_name).stem
        labels = list(info.get("labels", []))
        samples.append({"id": img_id, "stem": stem, "file_name": file_name, "labels": labels})
    return samples


def find_gt_path(gt_dir, stem, dataset_type):
    """Locate GT segmentation file for a given image stem."""
    if dataset_type == "voc":
        p = gt_dir / f"{stem}.png"
        return p if p.is_file() else None
    if dataset_type == "coco":
        p = gt_dir / f"{stem}.png"
        return p if p.is_file() else None
    if dataset_type == "ade20k":
        p = gt_dir / f"{stem}.png"
        return p if p.is_file() else None
    if dataset_type == "cityscapes":
        # gtFine/<split>/<city>/<stem>_gtFine_labelIds.png
        city = stem.split("_")[0]
        p = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        return p if p.is_file() else None
    return None


def find_img_path(img_dir, stem, dataset_type, file_name=None):
    """Locate original image file."""
    if dataset_type == "voc":
        p = img_dir / f"{stem}.jpg"
        return p if p.is_file() else None
    if dataset_type == "coco":
        p = img_dir / f"{stem}.jpg"
        return p if p.is_file() else None
    if dataset_type == "ade20k":
        p = img_dir / f"{stem}.jpg"
        return p if p.is_file() else None
    if dataset_type == "cityscapes":
        city = stem.split("_")[0]
        p = img_dir / city / f"{stem}_leftImg8bit.png"
        return p if p.is_file() else None
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pseudo, gt, num_classes, ignore_index=255):
    """Compute per-image metrics.

    Returns dict with:
      miou, miou_no_bg, pixel_acc, fg_pixel_acc,
      fg_precision, fg_recall, fg_f1,
      per_class_iou, fg_pixel_ratio, n_pseudo_classes, n_gt_classes
    """
    pseudo = np.asarray(pseudo).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)

    # Mask out ignore regions in GT
    if ignore_index is not None:
        valid = gt != ignore_index
    else:
        valid = np.ones_like(gt, dtype=bool)

    pseudo_v = pseudo[valid]
    gt_v = gt[valid]

    # Per-class IoU
    per_class_iou = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        pred_c = pseudo_v == c
        gt_c = gt_v == c
        union = pred_c | gt_c
        if union.sum() == 0:
            continue  # class absent in both -> skip (nan)
        inter = (pred_c & gt_c).sum()
        per_class_iou[c] = inter / union.sum()

    valid_classes = ~np.isnan(per_class_iou)
    miou = float(per_class_iou[valid_classes].mean()) if valid_classes.any() else 0.0
    # mIoU without background (class 0)
    fg_valid = valid_classes.copy()
    fg_valid[0] = False
    miou_no_bg = float(per_class_iou[fg_valid].mean()) if fg_valid.any() else 0.0

    # Pixel accuracy
    pixel_acc = float((pseudo_v == gt_v).sum()) / max(len(gt_v), 1)
    # Foreground pixel accuracy (GT foreground pixels correctly classified)
    gt_fg = gt_v > 0
    if gt_fg.sum() > 0:
        fg_pixel_acc = float((pseudo_v[gt_fg] == gt_v[gt_fg]).sum()) / gt_fg.sum()
    else:
        fg_pixel_acc = 1.0  # no foreground in GT -> trivially correct

    # Foreground binary metrics (any fg class vs background)
    pseudo_fg = pseudo_v > 0
    tp = float((pseudo_fg & gt_fg).sum())
    fp = float((pseudo_fg & ~gt_fg).sum())
    fn = float((~pseudo_fg & gt_fg).sum())
    fg_precision = tp / max(tp + fp, 1)
    fg_recall = tp / max(tp + fn, 1)
    fg_f1 = 2 * fg_precision * fg_recall / max(fg_precision + fg_recall, 1e-8)

    # Stats
    fg_pixel_ratio = float(pseudo_fg.sum()) / max(len(pseudo_v), 1)
    n_pseudo_classes = int(len(np.unique(pseudo_v[pseudo_v > 0])))
    n_gt_classes = int(len(np.unique(gt_v[gt_v > 0])))

    return {
        "miou": miou,
        "miou_no_bg": miou_no_bg,
        "pixel_acc": pixel_acc,
        "fg_pixel_acc": fg_pixel_acc,
        "fg_precision": fg_precision,
        "fg_recall": fg_recall,
        "fg_f1": fg_f1,
        "per_class_iou": per_class_iou,
        "fg_pixel_ratio": fg_pixel_ratio,
        "n_pseudo_classes": n_pseudo_classes,
        "n_gt_classes": n_gt_classes,
    }


def label_to_pixel_value(label, dataset_type):
    """Image-level label -> pseudo pixel value (mirror generator)."""
    if dataset_type in ("voc", "coco"):
        return int(label)  # label 1..C -> pixel 1..C
    return int(label) + 1  # ade/cityscapes: label 0..C-1 -> pixel 1..C


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

VOC_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def colorize_mask(mask, num_classes=21, ignore_index=255):
    """Convert class-index mask to RGB using VOC palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(VOC_PALETTE))):
        rgb[mask == c] = VOC_PALETTE[c]
    # ignore -> white
    rgb[mask == ignore_index] = [255, 255, 255]
    return rgb


def save_visualization(img_path, pseudo, gt, save_path, img_id, miou, dataset_type):
    """Save a side-by-side comparison: original | pseudo | GT | error map."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    pseudo_rgb = colorize_mask(pseudo, num_classes=21)
    gt_rgb = colorize_mask(gt, num_classes=21)

    # Error map: red = false positive (pseudo fg, gt bg), blue = false negative
    error = np.zeros((h, w, 3), dtype=np.uint8)
    valid = gt != 255
    pseudo_fg = (pseudo > 0) & valid
    gt_fg = (gt > 0) & valid
    error[pseudo_fg & ~gt_fg] = [255, 0, 0]    # red: over-segmentation
    error[~pseudo_fg & gt_fg] = [0, 0, 255]    # blue: under-segmentation
    error[(pseudo == gt) & valid & gt_fg] = [0, 200, 0]  # green: correct fg

    # Overlay pseudo on image
    alpha = 0.5
    overlay = img_rgb.copy()
    fg_mask = pseudo > 0
    overlay[fg_mask] = (overlay[fg_mask] * (1 - alpha) + pseudo_rgb[fg_mask] * alpha).astype(np.uint8)

    # Compose 2x2 grid
    pad = 4
    canvas_h = h * 2 + pad * 3
    canvas_w = w * 2 + pad * 3
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    def place(img, row, col, title):
        y0 = row * (h + pad) + pad
        x0 = col * (w + pad) + pad
        canvas[y0:y0 + h, x0:x0 + w] = img
        # title bar
        cv2.putText(canvas, title, (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    place(overlay, 0, 0, "Pseudo overlay")
    place(pseudo_rgb, 0, 1, "Pseudo mask")
    place(gt_rgb, 1, 0, "GT mask")
    place(error, 1, 1, f"Error (mIoU={miou:.3f})")

    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 pseudo mask quality against GT")
    parser.add_argument("--dataset", type=str, default="voc",
                        choices=["voc", "coco", "ade20k", "cityscapes"])
    parser.add_argument("--split", type=str, default="train",
                        help="Split to evaluate (default: train)")
    parser.add_argument("--pseudo-dir", type=str, default=None,
                        help="Override pseudo mask directory")
    parser.add_argument("--worst", type=int, default=20,
                        help="Number of worst samples to list (default: 20)")
    parser.add_argument("--vis", action="store_true",
                        help="Save visualization for worst-N samples")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Visualization output dir (default: pseudo_dir/../eval_vis)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of samples (0=all, for quick test)")
    args = parser.parse_args()

    os.environ["OVERRIDE_DATASET_TYPE"] = args.dataset
    from configs.config import NUM_CLASSES

    dataset_type = args.dataset
    root, pseudo_dir, gt_dir, img_dir, imglevel_json, cls_file, has_ignore = \
        get_dataset_paths(dataset_type, args.split, args.pseudo_dir)
    ignore_index = 255 if has_ignore else None
    num_classes = NUM_CLASSES

    print("=" * 80)
    print(f"Pseudo Mask Quality Evaluation - {dataset_type.upper()} ({args.split})")
    print("=" * 80)
    print(f"  Pseudo dir:    {pseudo_dir}")
    print(f"  GT dir:        {gt_dir}")
    print(f"  Image-level:   {imglevel_json}")
    print(f"  Num classes:   {num_classes}")
    print(f"  Ignore index:  {ignore_index}")

    class_names = load_class_names(cls_file, dataset_type)
    all_class_names = ["background"] + class_names

    samples = load_image_list(imglevel_json, dataset_type)
    print(f"  Total images in image-level json: {len(samples)}")

    # Filter: must have pseudo + GT
    eval_samples = []
    missing_pseudo = 0
    missing_gt = 0
    for s in samples:
        stem = s["stem"]
        p_path = pseudo_dir / f"{stem}.png"
        g_path = find_gt_path(gt_dir, stem, dataset_type)
        if not p_path.is_file():
            missing_pseudo += 1
            continue
        if g_path is None:
            missing_gt += 1
            continue
        eval_samples.append({**s, "pseudo_path": p_path, "gt_path": g_path})

    print(f"  Missing pseudo: {missing_pseudo}, missing GT: {missing_gt}")
    print(f"  Evaluable:      {len(eval_samples)}")

    if args.max_samples > 0:
        eval_samples = eval_samples[:args.max_samples]
        print(f"  Limited to:     {len(eval_samples)}")

    if len(eval_samples) == 0:
        print("\nNo evaluable samples found. Check paths.")
        return

    # ---- Per-image evaluation ----
    all_metrics = []
    per_class_inter = np.zeros(num_classes, dtype=np.float64)
    per_class_union = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    correct_pixels = 0
    fg_tp = fg_fp = fg_fn = 0
    empty_masks = 0
    label_violations = 0

    t0 = time.time()
    for s in tqdm(eval_samples, desc="Evaluating", ncols=100):
        pseudo = np.array(Image.open(s["pseudo_path"]))
        gt = np.array(Image.open(s["gt_path"]))

        # Resize if shapes mismatch (pseudo should match original image size)
        if pseudo.shape != gt.shape:
            pseudo = np.array(Image.fromarray(pseudo).resize(
                (gt.shape[1], gt.shape[0]), resample=Image.NEAREST))

        m = compute_metrics(pseudo, gt, num_classes, ignore_index)
        m["id"] = s["id"]
        m["stem"] = s["stem"]
        all_metrics.append(m)

        # Accumulate for global metrics
        if ignore_index is not None:
            valid = gt != ignore_index
        else:
            valid = np.ones_like(gt, dtype=bool)
        pv = pseudo[valid]
        gv = gt[valid]
        total_pixels += len(gv)
        correct_pixels += int((pv == gv).sum())

        for c in range(num_classes):
            pc = pv == c
            gc = gv == c
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

        pf = pv > 0
        gf = gv > 0
        fg_tp += (pf & gf).sum()
        fg_fp += (pf & ~gf).sum()
        fg_fn += (~pf & gf).sum()

        if m["n_pseudo_classes"] == 0:
            empty_masks += 1

        # Image-level label consistency: pseudo classes should all be in allowed labels
        allowed_pixels = set()
        for lab in s.get("labels", []):
            pv_val = label_to_pixel_value(lab, dataset_type)
            if 1 <= pv_val < num_classes:
                allowed_pixels.add(pv_val)
        pseudo_classes = set(int(v) for v in np.unique(pv) if v > 0)
        if not pseudo_classes.issubset(allowed_pixels):
            label_violations += 1

    elapsed = time.time() - t0

    # ---- Global metrics ----
    global_per_class_iou = np.full(num_classes, np.nan)
    for c in range(num_classes):
        if per_class_union[c] > 0:
            global_per_class_iou[c] = per_class_inter[c] / per_class_union[c]
    valid_cls = ~np.isnan(global_per_class_iou)
    global_miou = float(global_per_class_iou[valid_cls].mean()) if valid_cls.any() else 0.0
    fg_valid = valid_cls.copy()
    fg_valid[0] = False
    global_miou_no_bg = float(global_per_class_iou[fg_valid].mean()) if fg_valid.any() else 0.0
    global_pixel_acc = correct_pixels / max(total_pixels, 1)
    global_fg_precision = fg_tp / max(fg_tp + fg_fp, 1)
    global_fg_recall = fg_tp / max(fg_tp + fg_fn, 1)
    global_fg_f1 = 2 * global_fg_precision * global_fg_recall / max(
        global_fg_precision + global_fg_recall, 1e-8)

    # Per-image average metrics
    avg_miou = float(np.mean([m["miou"] for m in all_metrics]))
    avg_miou_no_bg = float(np.mean([m["miou_no_bg"] for m in all_metrics]))
    avg_pixel_acc = float(np.mean([m["pixel_acc"] for m in all_metrics]))
    avg_fg_precision = float(np.mean([m["fg_precision"] for m in all_metrics]))
    avg_fg_recall = float(np.mean([m["fg_recall"] for m in all_metrics]))
    avg_fg_f1 = float(np.mean([m["fg_f1"] for m in all_metrics]))

    fg_ratios = [m["fg_pixel_ratio"] for m in all_metrics]

    # ---- Report ----
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(f"  Samples evaluated:     {len(all_metrics)}")
    print(f"  Time:                  {elapsed:.1f}s")
    print()
    print("  --- Global (pixel-accumulated) ---")
    print(f"  mIoU (all classes):    {global_miou:.4f}")
    print(f"  mIoU (fg only):        {global_miou_no_bg:.4f}")
    print(f"  Pixel accuracy:        {global_pixel_acc:.4f}")
    print(f"  FG precision:          {global_fg_precision:.4f}")
    print(f"  FG recall:             {global_fg_recall:.4f}")
    print(f"  FG F1:                 {global_fg_f1:.4f}")
    print()
    print("  --- Per-image average ---")
    print(f"  avg mIoU (all):        {avg_miou:.4f}")
    print(f"  avg mIoU (fg):         {avg_miou_no_bg:.4f}")
    print(f"  avg pixel acc:         {avg_pixel_acc:.4f}")
    print(f"  avg FG precision:      {avg_fg_precision:.4f}")
    print(f"  avg FG recall:         {avg_fg_recall:.4f}")
    print(f"  avg FG F1:             {avg_fg_f1:.4f}")
    print()
    print("  --- Statistics ---")
    print(f"  Empty masks (all bg):  {empty_masks} / {len(all_metrics)} ({100*empty_masks/len(all_metrics):.1f}%)")
    print(f"  Label violations:      {label_violations} / {len(all_metrics)} ({100*label_violations/len(all_metrics):.1f}%)")
    print(f"  FG pixel ratio:        mean={np.mean(fg_ratios):.3f}, median={np.median(fg_ratios):.3f}, "
          f"p25={np.percentile(fg_ratios,25):.3f}, p75={np.percentile(fg_ratios,75):.3f}")
    print(f"  Pseudo classes/img:    mean={np.mean([m['n_pseudo_classes'] for m in all_metrics]):.1f}, "
          f"GT classes/img: mean={np.mean([m['n_gt_classes'] for m in all_metrics]):.1f}")

    print()
    print("  --- Per-class IoU (global) ---")
    print(f"  {'Class':<20s} {'IoU':>8s} {'Union%':>8s}")
    print("  " + "-" * 40)
    for c in range(num_classes):
        name = all_class_names[c] if c < len(all_class_names) else f"cls_{c}"
        iou = global_per_class_iou[c]
        union_pct = 100 * per_class_union[c] / max(total_pixels, 1)
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A "
        print(f"  {name:<20s} {iou_str:>8s} {union_pct:>7.2f}%")
    print("  " + "-" * 40)
    print(f"  {'Mean mIoU':<20s} {global_miou:>8.4f}")
    print(f"  {'Mean FG mIoU':<20s} {global_miou_no_bg:>8.4f}")

    # ---- Worst samples ----
    sorted_by_miou = sorted(all_metrics, key=lambda x: x["miou"])
    worst_n = sorted_by_miou[:args.worst]
    print(f"\n  --- Worst {len(worst_n)} samples by per-image mIoU ---")
    print(f"  {'Rank':>4s} {'ID':<20s} {'mIoU':>7s} {'mIoU_fg':>8s} {'PixAcc':>7s} {'FG_P':>6s} {'FG_R':>6s} {'#cls':>5s}")
    print("  " + "-" * 75)
    for rank, m in enumerate(worst_n, 1):
        print(f"  {rank:>4d} {m['stem']:<20s} {m['miou']:>7.4f} {m['miou_no_bg']:>8.4f} "
              f"{m['pixel_acc']:>7.4f} {m['fg_precision']:>6.3f} {m['fg_recall']:>6.3f} {m['n_pseudo_classes']:>5d}")

    # ---- Visualization ----
    if args.vis:
        vis_dir = Path(args.vis_dir) if args.vis_dir else pseudo_dir.parent / "eval_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving visualizations to: {vis_dir}")
        for rank, m in enumerate(worst_n, 1):
            stem = m["stem"]
            img_path = find_img_path(img_dir, stem, dataset_type)
            if img_path is None:
                continue
            pseudo = np.array(Image.open(pseudo_dir / f"{stem}.png"))
            gt_path = find_gt_path(gt_dir, stem, dataset_type)
            gt = np.array(Image.open(gt_path))
            if pseudo.shape != gt.shape:
                pseudo = np.array(Image.fromarray(pseudo).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST))
            save_path = vis_dir / f"{rank:03d}_{stem}_miou{m['miou']:.3f}.jpg"
            save_visualization(img_path, pseudo, gt, save_path, stem, m["miou"], dataset_type)
        print(f"  Saved {len(worst_n)} visualizations.")

    # ---- Save report ----
    report_dir = pseudo_dir.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"pseudo_quality_{dataset_type}_{args.split}_{ts}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Pseudo Mask Quality Report - {dataset_type.upper()} ({args.split})\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pseudo dir: {pseudo_dir}\n")
        f.write(f"Samples: {len(all_metrics)}\n\n")
        f.write(f"Global mIoU: {global_miou:.4f}\n")
        f.write(f"Global mIoU (fg): {global_miou_no_bg:.4f}\n")
        f.write(f"Pixel accuracy: {global_pixel_acc:.4f}\n")
        f.write(f"FG precision: {global_fg_precision:.4f}\n")
        f.write(f"FG recall: {global_fg_recall:.4f}\n")
        f.write(f"FG F1: {global_fg_f1:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for c in range(num_classes):
            name = all_class_names[c] if c < len(all_class_names) else f"cls_{c}"
            iou = global_per_class_iou[c]
            f.write(f"  {name}: {iou:.4f}\n" if not np.isnan(iou) else f"  {name}: N/A\n")
    print(f"\n  Report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
