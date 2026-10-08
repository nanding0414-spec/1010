"""CPU-only cleanup of cached binary proposals, without changing model scores."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from configs.experiment import CLEANUP


@dataclass(frozen=True)
class CleanupConfig:
    min_score: float = CLEANUP["min_score"]
    min_mask_area: int = CLEANUP["min_mask_area"]
    min_island_area: int = CLEANUP["min_island_area"]
    max_hole_area: int = CLEANUP["max_hole_area"]
    max_hole_fraction: float = CLEANUP["max_hole_fraction"]
    nms_iou: float = CLEANUP["nms_iou"]

    def __post_init__(self):
        if not 0 <= self.min_score <= 1 or not 0 <= self.nms_iou <= 1:
            raise ValueError("min_score and nms_iou must be in [0, 1]")
        if min(self.min_mask_area, self.min_island_area, self.max_hole_area) < 0:
            raise ValueError("Area thresholds must be nonnegative")
        if not 0 <= self.max_hole_fraction <= 1:
            raise ValueError("max_hole_fraction must be in [0, 1]")

    def to_dict(self):
        return asdict(self)


def _components(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8
    )
    return labels, stats[:, cv2.CC_STAT_AREA]


def _hole_ids(labels):
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return np.setdiff1d(np.unique(labels), np.append(border, 0))


def clean_mask(mask, config):
    """Remove small islands and fill only tiny enclosed holes; never erode edges."""
    mask = np.asarray(mask, dtype=bool)
    labels, areas = _components(mask)
    keep = areas >= config.min_island_area
    keep[0] = False
    result = keep[labels]
    removed = int(mask.sum() - result.sum())
    hole_limit = min(config.max_hole_area, int(result.sum() * config.max_hole_fraction))
    filled = 0
    if hole_limit > 0 and result.any():
        hole_labels, hole_areas = _components(~result)
        holes = _hole_ids(hole_labels)
        fill_ids = holes[hole_areas[holes] <= hole_limit]
        fill = np.zeros(len(hole_areas), dtype=bool)
        fill[fill_ids] = True
        filled = int(hole_areas[fill_ids].sum())
        result |= fill[hole_labels]
    return result, removed, filled


def mask_nms(masks, scores, threshold):
    """True-IoU deduplication. Containment alone does not suppress small objects."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    areas = [int(mask.sum()) for mask in masks]
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    kept = []
    for index in order:
        ax0, ay0, ax1, ay1 = boxes[index]
        duplicate = False
        for previous in kept:
            # The area ratio bounds IoU, including for nested object proposals.
            if min(areas[index], areas[previous]) <= threshold * max(areas[index], areas[previous]):
                continue
            bx0, by0, bx1, by1 = boxes[previous]
            x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if x0 >= x1 or y0 >= y1:
                continue
            intersection = np.count_nonzero(
                masks[index][y0:y1, x0:x1] & masks[previous][y0:y1, x0:x1]
            )
            union = areas[index] + areas[previous] - intersection
            if intersection > threshold * union:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
    return kept


def proposal_stats(masks):
    components = holes = 0
    for mask in masks:
        _, areas = _components(mask)
        components += len(areas) - 1
        labels, _ = _components(~mask.astype(bool))
        holes += len(_hole_ids(labels))
    return {
        "count": len(masks),
        "components": components,
        "holes": holes,
        "coverage": float(masks.any(axis=0).mean()) if len(masks) else 0.0,
    }


def clean_proposals(masks, scores, config):
    masks, scores = np.asarray(masks), np.asarray(scores, dtype=np.float32)
    if masks.ndim != 3 or min(masks.shape[1:]) < 1 or scores.shape != (len(masks),):
        raise ValueError("Expected masks [N,H,W] and scores [N]")
    if not np.isfinite(scores).all() or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Expected finite scores and binary masks")
    candidates, candidate_scores, source_indices = [], [], []
    report = dict(score_rejected=0, area_rejected=0, nms_rejected=0,
                  removed_island_pixels=0, filled_hole_pixels=0)
    for index, (mask, score) in enumerate(zip(masks, scores)):
        if score < config.min_score:
            report["score_rejected"] += 1
            continue
        result, removed, filled = clean_mask(mask, config)
        report["removed_island_pixels"] += removed
        report["filled_hole_pixels"] += filled
        if not result.any() or result.sum() < config.min_mask_area:
            report["area_rejected"] += 1
            continue
        candidates.append(result)
        candidate_scores.append(float(score))
        source_indices.append(index)
    keep = mask_nms(candidates, candidate_scores, config.nms_iou)
    report["nms_rejected"] = len(candidates) - len(keep)
    output = (np.stack([candidates[i] for i in keep]) if keep else
              np.zeros((0, *masks.shape[1:]), dtype=bool))
    return (output, np.asarray([candidate_scores[i] for i in keep], dtype=np.float32),
            np.asarray([source_indices[i] for i in keep], dtype=np.int64), report)
