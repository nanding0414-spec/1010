"""HCR: high-confidence retrieval rescue of discarded cached proposals.

No ground truth, learned parameters, or feature-space changes. Original clean
proposals take precedence; rescued pixels can only fill predicted background.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class HCRConfig:
    proposal_score: float = .4
    retrieval_score: float = .95
    margin: float = .1
    novel_fraction: float = .1
    min_area: int = 0
    dedup_iou: float = 1.
    extra_nms_iou: float = 1.
    class_thresholds: dict | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'class_thresholds':
                if value is not None and any(not math.isfinite(v) or v<0 or v>1.000001 for v in value.values()):
                    raise ValueError('Invalid class calibration thresholds')
                continue
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite HCR parameter: {name}')
        if not (0 <= self.proposal_score <= 1 and -1 <= self.retrieval_score <= 1
                and 0 <= self.margin <= 2 and 0 <= self.novel_fraction <= 1):
            raise ValueError('Invalid HCR thresholds')
        if self.min_area < 0 or int(self.min_area) != self.min_area or not 0 <= self.dedup_iou <= 1 or not 0 <= self.extra_nms_iou <= 1:
            raise ValueError('Invalid HCR geometry controls')


def mask_hash(mask):
    mask = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(str(mask.shape).encode() + mask.tobytes()).hexdigest()


def validate_masks(masks, scores):
    if len(masks) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid HCR masks/scores')
    if any(float(s) < 0 or float(s) > 1 for s in scores):
        raise ValueError('Proposal confidence must be in [0,1]')
    shape = None
    for mask in masks:
        array = np.asarray(mask)
        if array.ndim != 2 or not np.all((array == 0) | (array == 1)):
            raise ValueError('HCR masks must be binary 2D arrays')
        if shape is not None and shape != array.shape:
            raise ValueError('HCR mask dimensions differ')
        shape = array.shape
    return shape


def prepare_pool(strong_masks, strong_scores, raw_masks, raw_scores, cleanup, hcr=None):
    """Keep clean masks untouched; deduplicate relaxed raw candidates by geometry."""
    from src.proposal_cleanup import clean_proposals
    strong_shape = validate_masks(strong_masks, strong_scores)
    raw_shape = validate_masks(raw_masks, raw_scores)
    if strong_shape and raw_shape and strong_shape != raw_shape:
        raise ValueError('Clean/raw proposal dimensions differ')
    raw_array = np.asarray(raw_masks, dtype=bool)
    if not len(raw_array):
        raw_array = np.zeros((0, *(strong_shape or (1, 1))), dtype=bool)
    relaxed, scores, _, _ = clean_proposals(raw_array, raw_scores, replace(cleanup, min_score=0.))
    retained = [i for i,m in enumerate(strong_masks) if hcr is None or np.count_nonzero(m)>=hcr.min_area]
    masks = [np.asarray(strong_masks[i], dtype=bool) for i in retained]
    output_scores = [float(strong_scores[i]) for i in retained]
    flags = [False] * len(masks)
    seen = {mask_hash(m) for m in masks}
    for mask, score in zip(relaxed, scores):
        if hcr is not None:
            if mask.sum() < hcr.min_area:
                continue
            if hcr.dedup_iou < 1 and any(mask_iou(mask, other) >= hcr.dedup_iou for other in masks):
                continue
        digest = mask_hash(mask)
        if digest in seen:
            continue
        seen.add(digest)
        masks.append(mask.astype(bool))
        output_scores.append(float(score))
        flags.append(True)
    return masks, output_scores, flags


def mask_iou(a, b):
    return float(np.count_nonzero(a & b)) / max(np.count_nonzero(a | b), 1)


class RescueSource:
    """Read-only raw companion of a previously cleaned proposal cache."""
    def __init__(self, clean_dir, image_ids, raw_dir=None, hcr=None):
        from src.proposal_cleanup import CleanupConfig
        self.clean_dir = Path(clean_dir).resolve()
        metadata = self.clean_dir / 'cleanup_config.json'
        content = metadata.read_bytes()
        value = json.loads(content)
        self.cleanup = CleanupConfig(**value['config'])
        self.hcr = hcr
        self.raw_dir = Path(raw_dir or value['input_dir']).resolve()
        if self.raw_dir == self.clean_dir:
            raise ValueError('HCR requires a distinct raw proposal source')
        if value.get('errors', 0) != 0:
            raise ValueError('Clean proposal cache contains reported errors')
        for image_id in image_ids:
            for root in (self.clean_dir, self.raw_dir):
                if not (root / f'{image_id}.npz').is_file():
                    raise FileNotFoundError(root / f'{image_id}.npz')
        self.description = dict(module='HCR', version=1, raw_dir=str(self.raw_dir),
                                clean_dir=str(self.clean_dir), cleanup=self.cleanup.to_dict(),
                                cleanup_metadata_sha256=hashlib.sha256(content).hexdigest(),
                                relaxed_min_score=0., geometry='original_image',
                                insertion='background_only_after_original_foreground_NMS')

    def expand(self, image_id, masks, scores, height, width):
        raw_path = self.raw_dir / f'{image_id}.npz'
        with np.load(raw_path, allow_pickle=False) as data:
            raw, raw_scores = data['masks'], data['scores']
        if raw.ndim != 3 or raw_scores.shape != (len(raw),):
            raise ValueError(f'Invalid raw proposal arrays: {raw_path}')
        if raw.shape[1:] != (height, width) and not (len(raw) == 0 and raw.shape[1:] == (1, 1)):
            raise ValueError(f'Raw proposal/image dimensions differ: {raw_path}')
        output = prepare_pool(masks, scores, raw, raw_scores, self.cleanup, self.hcr)
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        identity = dict(image_id=image_id, clean_sha256=digest(self.clean_dir / f'{image_id}.npz'),
                        raw_sha256=digest(raw_path), strong_count=len(masks), extra_count=sum(output[2]))
        return (*output, identity)


def rescue_instances(strong, extras, config, foreground_classes, stats=None):
    """Accept extras after strong-only NMS; no extra NMS or foreground overwrite."""
    stats = stats if stats is not None else {}
    stats.update(extra_candidates=len(extras), eligible=0, accepted=0, added_pixels=0)
    if not strong and not extras:
        return []
    foreground = set(foreground_classes)
    shape = (strong or extras)[0]['mask'].shape
    occupied = np.zeros(shape, dtype=bool)
    for row in [*strong, *extras]:
        if row['mask'].shape != shape or row['mask'].dtype != bool:
            raise ValueError('Invalid HCR instance mask')
        if any(not math.isfinite(float(row[k])) for k in
               ('score', 'proposal_score', 'retrieval_score', 'retrieval_margin')):
            raise ValueError('Nonfinite HCR instance score')
    for row in strong:
        if row['cls_name'] in foreground:
            occupied |= row['mask']
    result = list(strong)
    accepted_masks = {}
    for row in sorted(extras, key=lambda r: -r['score']):
        threshold = (config.class_thresholds.get(row['cls_name'], 1.000001)
                     if config.class_thresholds is not None else config.retrieval_score)
        if (row['cls_name'] not in foreground or row['proposal_score'] < config.proposal_score
                or row['retrieval_score'] < threshold
                or row['retrieval_margin'] < config.margin
                or row['mask'].sum() < config.min_area):
            continue
        if config.extra_nms_iou < 1 and any(mask_iou(row['mask'], other) >= config.extra_nms_iou
                for other in accepted_masks.get(row['cls_name'], [])):
            continue
        stats['eligible'] += 1
        added = row['mask'] & ~occupied
        if not added.any() or added.sum() / max(int(row['mask'].sum()), 1) < config.novel_fraction:
            continue
        result.append(dict(row, mask=added))
        accepted_masks.setdefault(row['cls_name'], []).append(row['mask'])
        occupied |= added
        stats['accepted'] += 1
        stats['added_pixels'] += int(added.sum())
    return result
