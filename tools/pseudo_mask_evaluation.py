"""Evaluate saved pseudo-label PNGs without loading SAM3 or retrieval code."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


CITY_LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)


def normalize_split(split, dataset):
    train, val = ('training', 'validation') if dataset == 'ade20k' else ('train', 'val')
    return train if split in ('train', 'training') else val if split in ('val', 'validation') else split


def find_ground_truth(image, root, dataset, split, metadata=None, gt_dir=None, gt_format='auto'):
    image, root = Path(image), Path(root)
    filename = image.stem + '.png'
    if dataset == 'cityscapes':
        base = image.stem.removesuffix('_leftImg8bit')
        filenames = [f'{base}_gtFine_labelIds.png', f'{base}_gtFine_labelTrainIds.png', filename]
        if gt_format == 'trainids':
            filenames = [f'{base}_gtFine_labelTrainIds.png', filename]
        elif gt_format == 'labelids':
            filenames = [f'{base}_gtFine_labelIds.png', filename]
        directory = Path(gt_dir) if gt_dir else root / 'gtFine' / normalize_split(split, dataset)
        candidates = [directory / image.parent.name / name for name in filenames]
        candidates += [directory / name for name in filenames]
    elif gt_dir:
        candidates = [Path(gt_dir) / filename]
    elif dataset == 'voc':
        folders = ['SegmentationClassAug', 'SegmentationClass'] if split == 'trainaug' else ['SegmentationClass', 'SegmentationClassAug']
        candidates = [root / folder / filename for folder in folders]
    elif dataset == 'coco':
        candidates = [root / 'SegmentationClass' / split / filename]
    elif dataset == 'ade20k':
        candidates = [root / 'annotations' / normalize_split(split, dataset) / filename]
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    if not gt_dir and metadata and metadata.get('gt_file_name'):
        candidates.insert(0, root / metadata['gt_file_name'])
    return next((path for path in candidates if path.is_file()), None), candidates


def gt_to_internal(gt, dataset, path=None, gt_format='auto'):
    """Use 0=background and 1..C=foreground, 255=GT ignore for all datasets.

    Background predictions on valid ADE/City pixels remain errors, not ignored.
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError('Ground truth must be a 2-D integer class-ID image, not RGB')
    gt = gt.astype(np.int64)
    if gt_format == 'internal':
        return gt
    if dataset == 'cityscapes':
        if path and ((gt_format == 'labelids' and 'labelTrainIds' in str(path)) or
                     (gt_format == 'trainids' and 'labelIds' in str(path))):
            raise ValueError('Requested gt-format conflicts with Cityscapes GT filename')
        if gt_format == 'auto':
            gt_format = 'trainids' if path and 'labelTrainIds' in str(path) else 'labelids'
        if gt_format not in ('labelids', 'trainids'):
            raise ValueError('Cityscapes gt-format must be auto, labelids, trainids or internal')
        out = np.full(gt.shape, 255, dtype=np.int64)
        if gt_format == 'trainids':
            valid = (gt >= 0) & (gt < 19)
            out[valid] = gt[valid] + 1
        else:
            for internal, raw in enumerate(CITY_LABEL_IDS, 1):
                out[gt == raw] = internal
        return out
    if gt_format != 'auto':
        raise ValueError('labelids/trainids formats are only valid for Cityscapes')
    if dataset == 'ade20k':
        gt[gt == 0] = 255
    return gt


def accumulate_confusion(confusion, pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    count = len(confusion)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError(f'Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}; no resizing is applied')
    if not np.issubdtype(pred.dtype, np.integer) or np.any((pred < 0) | (pred >= count)):
        raise ValueError(f'Pseudo-label PNG must use IDs 0..{count - 1}')
    valid = gt != 255
    if np.any((gt[valid] < 0) | (gt[valid] >= count)):
        raise ValueError(f'GT contains class IDs outside 0..{count - 1} / 255')
    pairs = gt[valid].astype(np.int64) * count + pred[valid].astype(np.int64)
    confusion += np.bincount(pairs, minlength=count * count).reshape(count, count)
    return int(valid.sum())


def summarize_confusion(confusion, class_names, dataset):
    names = ['background'] + list(class_names)
    tp = confusion.diagonal().astype(np.float64)
    gt_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    union = gt_pixels + predicted_pixels - tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    main_ids = np.arange(len(names)) if dataset in ('voc', 'coco') else np.arange(1, len(names))

    def mean(ids):
        values = iou[ids]
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None

    return dict(
        mIoU=mean(main_ids), foreground_mIoU=mean(np.arange(1, len(names))),
        includes_background=dataset in ('voc', 'coco'),
        pixel_accuracy=float(tp.sum() / confusion.sum()) if confusion.sum() else None,
        valid_pixels=int(confusion.sum()),
        per_class=[dict(id=i, name=name, IoU=float(iou[i]) if np.isfinite(iou[i]) else None,
                        intersection=int(tp[i]), union=int(union[i]), gt_pixels=int(gt_pixels[i]),
                        predicted_pixels=int(predicted_pixels[i])) for i, name in enumerate(names)],
        absent_class_policy='Exclude classes with zero union; predicted-only classes have IoU 0',
        ignore_index=255, confusion_matrix=confusion.tolist())


def evaluate_pseudo_masks(samples, root, dataset, split, class_names, report_path,
                          gt_dir=None, gt_format='auto', resize_prediction_to_gt=False):
    """Samples: image, prediction, optional metadata and generation_failed keys."""
    confusion = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=np.int64)
    evaluated, missing_gt, missing_predictions, failures, no_valid_gt = [], [], [], [], []
    resized_predictions = []
    for sample in samples:
        image, prediction = Path(sample['image']), Path(sample['prediction'])
        if sample.get('generation_failed'):
            missing_predictions.append(str(image))
            continue  # Never score a stale PNG left by a failed regeneration.
        if not prediction.is_file():
            missing_predictions.append(str(image))
            continue
        gt_path, candidates = find_ground_truth(image, root, dataset, split, sample.get('metadata'), gt_dir, gt_format)
        if gt_path is None:
            missing_gt.append(dict(image=str(image), searched=[str(p) for p in candidates]))
            continue
        try:
            with Image.open(prediction) as source:
                pred = np.array(source)
            with Image.open(gt_path) as source:
                gt = gt_to_internal(np.array(source), dataset, gt_path, gt_format)
            if resize_prediction_to_gt:
                if (pred.ndim != 2 or not np.issubdtype(pred.dtype, np.integer)
                        or np.any((pred < 0) | (pred > len(class_names)))):
                    raise ValueError('Prediction must contain valid integer class IDs before resizing')
                with Image.open(image) as source:
                    if source.size != (pred.shape[1], pred.shape[0]):
                        raise ValueError('Saved prediction must match input image dimensions in resize-for-evaluation mode')
                if pred.shape != gt.shape:
                    if pred.shape[0] * gt.shape[1] != pred.shape[1] * gt.shape[0]:
                        raise ValueError('Prediction/GT aspect ratios differ')
                    resized_predictions.append(dict(prediction=str(prediction),
                        from_shape=list(pred.shape), to_shape=list(gt.shape)))
                    pred = np.asarray(Image.fromarray(pred.astype(np.int32)).resize(
                        (gt.shape[1], gt.shape[0]), resample=Image.Resampling.NEAREST))
            pixels = accumulate_confusion(confusion, pred, gt)
            if pixels:
                evaluated.append(dict(image=str(image), prediction=str(prediction), gt=str(gt_path)))
            else:
                no_valid_gt.append(str(image))
        except (OSError, ValueError) as exc:
            failures.append(dict(image=str(image), error=str(exc)))
    report = summarize_confusion(confusion, class_names, dataset)
    report.update(dataset=dataset, split=split, requested_count=len(samples), evaluated_count=len(evaluated),
                  evaluated=evaluated, missing_gt=missing_gt, missing_predictions=missing_predictions,
                  errors=failures, no_valid_gt=no_valid_gt,
                  prediction_alignment='nearest_to_gt_in_memory' if resize_prediction_to_gt else 'strict',
                  resized_predictions=resized_predictions,
                  status='complete' if len(evaluated) == len(samples) and evaluated else 'partial' if evaluated else 'unavailable',
                  scope='Pseudo-label quality on this split using image-level allowed classes; not end-to-end segmentation')
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if report['mIoU'] is None:
        print(f'[mIoU] unavailable: no valid GT/prediction pairs. Report: {report_path}')
    else:
        print(f'[mIoU] {100 * report["mIoU"]:.2f}% ({report["status"]}, {len(evaluated)}/{len(samples)} images)')
        print(f'[mIoU] Report: {report_path}')
    return report
