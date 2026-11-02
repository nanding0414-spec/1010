"""Explicit experimental clustering, pseudo-label losses, and calibration.

Calibration consumes held-out TRAIN pseudo regions, never final validation GT.
"""
import numpy as np


def spherical_centers(x, count=4, weights=None, iterations=12):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or count < 1:
        raise ValueError('Invalid prototype inputs')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    w = np.ones(len(x), np.float32) if weights is None else np.asarray(weights, np.float32)
    if w.shape != (len(x),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Prototype weights must be finite and positive')
    # Fit on a deterministic bounded sample; assign every member below.
    subset = np.linspace(0, len(x)-1, min(len(x), 512), dtype=int)
    fit = x[subset]; fw = w[subset]
    mean = (fit * fw[:, None]).sum(0); mean /= max(np.linalg.norm(mean), 1e-8)
    chosen = [int(np.argmax(fit @ mean))]
    while len(chosen) < min(count, len(fit)):
        similarity = (fit @ fit[chosen].T).max(1)
        idx = int(np.argmin(similarity))
        if similarity[idx] > 1-1e-6: break
        chosen.append(idx)
    centers = fit[chosen].copy()
    for _ in range(iterations):
        assignment = np.argmax(fit @ centers.T, axis=1)
        updated = []
        for k, center in enumerate(centers):
            rows = assignment == k
            value = (fit[rows] * fw[rows, None]).sum(0) if rows.any() else center
            updated.append(value / max(np.linalg.norm(value), 1e-8))
        updated = np.asarray(updated, np.float32)
        if np.allclose(centers, updated, atol=1e-5): break
        centers = updated
    # Final weighted means use all gallery members, not just the fit sample.
    assignment = np.argmax(x @ centers.T, axis=1)
    output = []
    for k in range(len(centers)):
        rows = assignment == k
        if rows.any():
            center = (x[rows] * w[rows, None]).sum(0)
            output.append(center / max(np.linalg.norm(center), 1e-8))
    return np.ascontiguousarray(output, dtype=np.float32)


def split_background(mask, tokens, count=4, min_area=300):
    """Partition background using cached token appearance; preserve every pixel.

    These are feature clusters, not reconstructed SAM instances.
    """
    from src.hard_center_gmp import select_centers
    mask = np.asarray(mask, bool)
    if mask.sum() < min_area: return []
    indices, _ = select_centers(mask)
    if len(indices) < count: return [mask.copy()]
    x = np.asarray(tokens, np.float32).reshape(72*72, -1)
    centers = spherical_centers(x[indices], count)
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    assignments = np.argmax(normalized @ centers.T, axis=1).reshape(72,72)
    h, w = mask.shape
    iy = np.minimum((np.arange(h)+.5)*72/h, 71).astype(int)
    ix = np.minimum((np.arange(w)+.5)*72/w, 71).astype(int)
    labels = assignments[iy[:,None], ix[None,:]]
    sizes = np.bincount(labels[mask], minlength=len(centers))
    keep = np.flatnonzero(sizes >= min_area)
    if len(keep) < 2: return [mask.copy()]
    for k in np.flatnonzero(sizes < min_area):
        target = keep[np.argmax(centers[keep] @ centers[k])]
        labels[labels == k] = target
    result = [mask & (labels == k) for k in keep]
    if not np.array_equal(np.sum(result, axis=0), mask.astype(int)):
        raise RuntimeError('Background partition lost or duplicated pixels')
    return result


def pseudo_classification_loss(logits, labels, kind='ce', q=.7, weights=None, smoothing=0.):
    import torch
    import torch.nn.functional as F
    if kind == 'ce':
        loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=smoothing)
    elif kind == 'gce':
        if not 0 < q <= 1 or smoothing:
            raise ValueError('GCE requires 0<q<=1 and no label smoothing')
        logp = F.log_softmax(logits, dim=1).gather(1, labels[:,None]).squeeze(1)
        loss = -torch.expm1(q*logp) / q
    else: raise ValueError('Unknown pseudo-label loss')
    if weights is None: return loss.mean()
    weights = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
    if weights.shape != loss.shape or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Invalid sample weights')
    return (loss*weights).sum()/weights.sum()


def probabilities(scores, temperature):
    if not np.isfinite(temperature) or temperature <= 0: raise ValueError('Invalid temperature')
    values = np.asarray(scores, np.float64)/temperature
    values -= values.max(axis=1, keepdims=True)
    result = np.exp(values)
    return result/result.sum(axis=1, keepdims=True)


def fit_calibration(scores, targets, labels, precision=.99, min_samples=20):
    scores = np.asarray(scores, np.float32); targets = np.asarray(targets, int)
    if (scores.shape != (len(targets), len(labels)) or not len(targets)
            or not np.isfinite(scores).all() or (targets<0).any() or (targets>=len(labels)).any()):
        raise ValueError('Invalid held-out calibration data')
    temperatures = np.geomspace(.005, 1., 100)
    nll = [float(-np.log(np.maximum(probabilities(scores,t)[np.arange(len(targets)),targets],1e-15)).mean())
           for t in temperatures]
    index = int(np.argmin(nll)); temperature = float(temperatures[index])
    p = probabilities(scores, temperature); predicted = p.argmax(1); confidence = p.max(1)
    thresholds = {}; support = {}
    for k, label in enumerate(labels):
        rows = np.flatnonzero(predicted == k)
        rows = rows[np.argsort(-confidence[rows], kind='stable')]
        correct = np.cumsum(targets[rows] == k)
        counts = np.arange(1,len(rows)+1)
        # Evaluate complete tie groups so inference's >= cutoff has the same support.
        last = np.r_[confidence[rows][:-1] != confidence[rows][1:], True] if len(rows) else np.array([],bool)
        valid = np.flatnonzero((counts>=min_samples) & (correct/np.maximum(counts,1)>=precision) & last)
        end = int(valid[-1]) if len(valid) else None
        thresholds[label] = float(confidence[rows[end]]) if end is not None else 1.000001
        support[label] = dict(predicted=len(rows), accepted=end+1 if end is not None else 0,
                              empirical_precision=float(correct[end]/(end+1)) if end is not None else None)
    return dict(temperature=temperature,nll=nll[index],nll_temperature1=nll[-1],
                thresholds=thresholds,support=support,target_precision=precision,min_samples=min_samples,
                source='held_out_train_images_pseudo_regions_not_GT',generalization_guarantee=False)
