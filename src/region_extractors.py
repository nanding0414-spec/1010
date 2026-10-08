"""Region extraction utilities for feature bank building."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from configs.config import (
    DATASET_TYPE,
    DATASET_CLASSES,
    MIN_MASK_AREA,
    BUILD_SPLIT_CONNECTED_COMPONENTS,
    PSEUDO_MASK_ROOT,
)


def _extract_class_regions(seg_map: np.ndarray, min_area: int):
    """Extract per-class binary masks from a segmentation map."""
    regions = []
    for cid in range(1, len(DATASET_CLASSES) + 1):
        mask_c = (seg_map == cid).astype(np.uint8)
        if not BUILD_SPLIT_CONNECTED_COMPONENTS:
            if mask_c.sum() >= min_area:
                regions.append({"cls_id": cid, "mask": mask_c})
            continue

        # Keep disconnected instances/modes separate. Pooling all same-class
        # objects in an image produces an averaged gallery vector that does not
        # match the instance masks used during online retrieval.
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask_c, 8)
        for component_id in range(1, n_components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            regions.append({
                "cls_id": cid,
                "mask": (labels == component_id).astype(np.uint8),
            })

    if DATASET_TYPE in ("ade20k", "cityscapes"):
        return regions

    bg = (seg_map == 0).astype(np.uint8)
    if bg.sum() >= min_area:
        regions.append({"cls_id": 0, "mask": bg})
    return regions


def get_regions_from_pseudo(img_path: str, min_area: int = MIN_MASK_AREA, pseudo_root=None):
    """Load pre-generated pseudo masks and split into per-class regions."""
    from configs.config import DATASET_TRAIN_SPLIT

    img_path = Path(img_path)
    pseudo_root = Path(PSEUDO_MASK_ROOT if pseudo_root is None else pseudo_root)

    if DATASET_TYPE == "coco":
        pseudo_path = pseudo_root / DATASET_TRAIN_SPLIT / f"{img_path.name.replace('.jpg', '.png')}"
    elif DATASET_TYPE in ("ade20k", "cityscapes"):
        pseudo_path = pseudo_root / DATASET_TRAIN_SPLIT / f"{img_path.stem}.png"
    else:
        pseudo_path = pseudo_root / f"{img_path.stem}.png"

    if not pseudo_path.is_file():
        raise FileNotFoundError(
            f"Pseudo mask not found: {pseudo_path}\n"
            f"Run: python tools/generate_pseudo_masks_sam3.py --dataset {DATASET_TYPE}"
        )

    pseudo_mask = np.array(Image.open(pseudo_path))
    return _extract_class_regions(pseudo_mask, min_area)


__all__ = [
    "get_regions_from_pseudo",
]
