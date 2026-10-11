"""Apply proposal cleanup geometry to semantic PNGs without inventing scores."""
import numpy as np

from src.proposal_cleanup import CleanupConfig, clean_mask

POLICY = 'semantic_per_class_geometry_only_no_scores_no_NMS_background_only_hole_fill'


def clean_semantic_labels(labels, config=None, num_classes=21):
    config = config or CleanupConfig()
    labels = np.asarray(labels)
    if (labels.ndim != 2 or min(labels.shape) < 1
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, range(num_classes)).all()):
        raise ValueError('Expected a semantic class-ID image, with values 0..C-1')
    result = np.zeros(labels.shape, np.uint8)
    channels = []
    report = dict(removed_island_pixels=0, proposed_hole_pixels=0,
                  filled_background_pixels=0, protected_other_class_pixels=0,
                  area_rejected_classes=0, classes=[])
    for cid in range(1, num_classes):
        mask = labels == cid
        if not mask.any():
            continue
        cleaned, removed, filled = clean_mask(mask, config)
        rejected = int(cleaned.sum()) < config.min_mask_area
        if rejected:
            cleaned[:] = False
        channels.append((cid, mask, cleaned))
        report['removed_island_pixels'] += removed
        report['proposed_hole_pixels'] += filled
        report['area_rejected_classes'] += int(rejected)
        report['classes'].append(dict(class_id=cid, before=int(mask.sum()),
            after_geometry=int(cleaned.sum()), removed_island_pixels=removed,
            proposed_hole_pixels=filled, area_rejected=rejected))
    # Preserve every surviving foreground class before claiming background holes.
    # A hole belonging to another input class must never be overwritten.
    for cid, original, cleaned in channels:
        result[original & cleaned] = cid
    for cid, original, cleaned in channels:
        additions = cleaned & ~original
        blocked = additions & (labels != 0)
        eligible = additions & (labels == 0) & (result == 0)
        report['protected_other_class_pixels'] += int(blocked.sum())
        report['filled_background_pixels'] += int(eligible.sum())
        result[eligible] = cid
    for row in report['classes']:
        row['after_composite'] = int((result == row['class_id']).sum())
    report['changed_pixels'] = int((result != labels).sum())
    report['changed'] = bool(report['changed_pixels'])
    return result, report


def clean_label_file(source, target, config, code_digest, expected_source=None):
    """Save a new semantic PNG and bind its checkpoint to the unchanged input."""
    import os
    from pathlib import Path
    from PIL import Image
    from tools.run_mainline_full import identity, checked_checkpoint, read, write
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Choose a new output path outside the source labels')
    original = identity(source)
    if expected_source is not None and original != expected_source:
        raise ValueError(f'Inherited label changed: {source}')
    binding = dict(source=original, config=config.to_dict(), policy=POLICY, code=code_digest)
    if not checked_checkpoint(target, binding):
        with Image.open(source) as image:
            cleaned, stats = clean_semantic_labels(np.asarray(image), config)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.png')
        Image.fromarray(cleaned).save(temporary); os.replace(temporary, target)
        write(target.with_suffix('.json'), dict(binding=binding, artifact=identity(target), statistics=stats))
    return read(target.with_suffix('.json'))['statistics']
