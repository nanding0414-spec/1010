#!/usr/bin/env python3
"""
Generate VOC2012 image-level weak-supervision labels.

Reads SegmentationClassAug masks and extracts per-image class indices.
Mask pixel values map directly to class indices in cls_voc21.txt (0=bg, 1-20=fg).
Value 255 is ignored.

Output:
    <VOC_ROOT>/ImageSets/ImageLevel/train_imagelevel.json
    <VOC_ROOT>/ImageSets/ImageLevel/val_imagelevel.json

Usage:
    python tools/generate_voc_imagelevel_train_val.py --voc-root data/VOC2012
    python tools/generate_voc_imagelevel_train_val.py --voc-root data/VOC2012 --splits trainaug
"""

import argparse
import os
import os.path as osp
import json
from typing import List
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Configurable paths (modify if your layout differs)
# ---------------------------------------------------------------------------
DEFAULT_VOC_ROOT = 'data/VOC2012'
DEFAULT_CLASS_CONFIG = 'configs/cls_voc21.txt'
SEG_MASK_DIR = 'SegmentationClassAug'          # GT mask directory under VOC_ROOT
SPLIT_DIR = 'ImageSets/Segmentation'            # split list directory
OUTPUT_DIR = 'ImageSets/ImageLevel'             # output directory under VOC_ROOT
IGNORE_VALUE = 255

VOC_CLASSES: List[str] = []


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Generate VOC2012 image-level label JSON')
    ap.add_argument('--voc-root', default=DEFAULT_VOC_ROOT, help='VOC2012 dataset root')
    ap.add_argument('--config-path', default=DEFAULT_CLASS_CONFIG, help='Class definition file')
    ap.add_argument('--splits', nargs='+', choices=['train', 'trainaug', 'val'],
                    default=['train', 'val'], help='Generate JSON only for these exact split lists')
    ap.add_argument('--seg-mask-dir', default=SEG_MASK_DIR,
                    help='Semantic annotation directory, relative to VOC root or absolute')
    return ap.parse_args(argv)


def load_classes(config_path: str) -> List[str]:
    """Load class names from config file (canonical first field per line)."""
    if not osp.isfile(config_path):
        raise FileNotFoundError(f'Class config not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        classes.append(line.split(';')[0].strip())
    return classes


def read_seg_ids(voc_root: str, split: str) -> List[str]:
    """Read image IDs from ImageSets/Segmentation/{split}.txt."""
    list_path = osp.join(voc_root, SPLIT_DIR, f'{split}.txt')
    if not osp.isfile(list_path):
        raise FileNotFoundError(f'Split list not found: {list_path}')
    with open(list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def extract_labels_from_mask(mask_path: str) -> List[int]:
    """Extract valid class indices from a PNG mask (ignoring 255)."""
    if not osp.isfile(mask_path):
        raise FileNotFoundError(f"Annotation mask not found: {mask_path}; cannot derive image-level labels")
    with Image.open(mask_path) as image:
        mask = np.array(image)
    values = np.unique(mask)
    num_classes = len(VOC_CLASSES)
    return sorted([int(v) for v in values if v != IGNORE_VALUE and 0 <= v < num_classes])


def build_payload(voc_root: str, split: str, ids: List[str], seg_mask_dir=SEG_MASK_DIR):
    """Build JSON payload for a given split."""
    images = []
    print(f"Processing {split} set ({len(ids)} images)...")
    for i, img_id in enumerate(ids):
        mask_path = osp.join(voc_root, seg_mask_dir, f'{img_id}.png')
        labels = extract_labels_from_mask(mask_path)
        images.append({
            'img_id': img_id,
            'file_name': f'JPEGImages/{img_id}.jpg',
            'labels': labels
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ids)}")
    return {'data_root': voc_root, 'images': images, 'classes': VOC_CLASSES}


def main(argv=None):
    args = parse_args(argv)
    voc_root = args.voc_root

    global VOC_CLASSES
    VOC_CLASSES = load_classes(args.config_path)
    print(f"Loaded {len(VOC_CLASSES)} classes from {args.config_path}")

    out_root = osp.join(voc_root, OUTPUT_DIR)
    os.makedirs(out_root, exist_ok=True)

    for split in dict.fromkeys(args.splits):
        ids = read_seg_ids(voc_root, split)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{split}.txt must contain nonempty, unique image IDs")
        if split == 'trainaug':
            val_path = osp.join(voc_root, SPLIT_DIR, 'val.txt')
            if osp.isfile(val_path):
                overlap = set(ids) & set(read_seg_ids(voc_root, 'val'))
                if overlap:
                    raise ValueError(f"trainaug includes {len(overlap)} validation images: {sorted(overlap)[:5]}")
            else:
                print("Warning: val.txt missing; trainaug/validation overlap was not checked")
        payload = build_payload(voc_root, split, ids, args.seg_mask_dir)
        out_path = osp.join(out_root, f'{split}_imagelevel.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(payload['images'])} entries -> {out_path}")


if __name__ == '__main__':
    main()
