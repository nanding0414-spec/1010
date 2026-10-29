"""Preregistered original-view controls. Pure NumPy; no crop scores, GT or IO."""
from __future__ import annotations
import numpy as np

PROTOCOL = 'r3_crossview_mechanism_control_v1'
SEEDS = (42, 123, 2026)
METHODS = ('A', 'D', 'E', 'E_count', 'E_area')
NEW_METHODS = ('E', 'E_count', 'E_area')
TARGETS = {42: dict(count=49, area=318574), 123: dict(count=50, area=444479),
           2026: dict(count=53, area=540285)}
RULES = dict(proposal=.4, original_score=.90, original_margin=.05, novel=.10,
             original_score_weight=.8, proposal_weight=.2, centroid_weight=.2,
             global_order='FP32 fused original confidence descending; image_id, candidate_id ties',
             count='first K actual sequential acceptances; shortage is an error',
             area='closest cumulative actual-acceptance prefix including zero; ties choose shorter',
             labels='original top1 foreground only', pixels='live background after A and earlier extras',
             crop_semantics_used_by_controls=False, new_training=0, new_SAM3_forward=0)


def eligibility(labels, scores, margins, proposals, ids, accepted_A):
    labels, scores, margins, proposals = map(np.asarray, (labels, scores, margins, proposals))
    n = len(ids)
    if any(x.shape != (n,) for x in (labels, scores, margins, proposals)):
        raise ValueError('Candidate rows differ')
    if not all(np.isfinite(x).all() for x in (scores, margins, proposals)):
        raise ValueError('Nonfinite original confidence')
    return ((labels > 0) & (labels < 21) & (proposals.astype(np.float64) >= .4)
            & (scores.astype(np.float64) >= .90) & (margins.astype(np.float64) >= .05)
            & ~np.isin(np.asarray(ids), list(accepted_A)))


def closest_prefix(added_areas, target):
    areas = np.asarray(added_areas, np.int64)
    if target < 0 or (areas <= 0).any():
        raise ValueError('Positive actual additions and nonnegative target required')
    cumulative = np.r_[np.int64(0), np.cumsum(areas)]
    k = int(np.argmin(np.abs(cumulative - int(target))))
    return k, int(cumulative[k]), int(abs(cumulative[k] - target))


def compose(images, count_target, area_target):
    """Global prefix has the same within-image order as ordinary E composition.

    images rows contain only A, mapped masks, original labels/scores/margins,
    proposal scores, candidate identities and A accepted identities.
    """
    predictions, owners, ordinals, eligible, traces = {}, {}, {}, {}, {}
    order = []
    seen = set()
    for stem in sorted(images):
        d = images[stem]; a = np.asarray(d['A']); masks = np.asarray(d['masks'])
        ids = np.asarray(d['candidate_ids']); n = len(ids)
        if a.ndim != 2 or masks.dtype != np.bool_ or masks.shape != (n, *a.shape):
            raise ValueError('Mask/A alignment differs')
        if len(set(ids.tolist())) != n or not np.isin(a, np.arange(21)).all():
            raise ValueError('Invalid class/identity')
        e = eligibility(d['labels'], d['scores'], d['margins'], d['proposals'], ids, d['accepted_A'])
        rank = .8 * np.asarray(d['scores'], np.float32) + .2 * np.asarray(d['proposals'], np.float32)
        predictions[stem] = a.copy(); owners[stem] = np.full(a.shape, -1, np.int32)
        ordinals[stem] = np.full(n, np.iinfo(np.int32).max, np.int32)
        eligible[stem] = ids[e].tolist(); traces[stem] = []
        for i in np.flatnonzero(e):
            key = (stem, str(ids[i]))
            if key in seen: raise ValueError('Duplicate global candidate identity')
            seen.add(key); order.append((float(rank[i]), stem, str(ids[i]), int(i)))
    order.sort(key=lambda x: (-x[0], x[1], x[2]))
    accepted = []
    for score, stem, cid, i in order:
        d = images[stem]; mask = d['masks'][i]; area = int(mask.sum())
        new = mask & (predictions[stem] == 0); added = int(new.sum())
        novel = added / max(area, 1); accept = added > 0 and novel >= .1
        trace = dict(image_id=stem, candidate_id=cid, index=i, fused_score=score,
                     added_pixels=added if accept else 0, available_pixels=added,
                     mask_area=area, novel_fraction=novel, accepted=accept)
        traces[stem].append(trace)
        if accept:
            predictions[stem][new] = int(d['labels'][i]); owners[stem][new] = i
            ordinals[stem][i] = len(accepted) + 1
            accepted.append(dict(trace, acceptance_ordinal=len(accepted) + 1))
    if len(accepted) < count_target:
        raise ValueError(f'Count match impossible: only {len(accepted)} actual acceptances, target {count_target}')
    area_k, area, error = closest_prefix([r['added_pixels'] for r in accepted], area_target)
    return dict(predictions=predictions, owners=owners, ordinals=ordinals,
                eligible=eligible, traces=traces, accepted=accepted, count_prefix=int(count_target),
                area_prefix=area_k, area_pixels=area, area_abs_error=error,
                total_added_pixels=sum(r['added_pixels'] for r in accepted))


def prefix(A, E, owner, ordinals, k):
    selected = owner >= 0
    if selected.any():
        selected[selected] = ordinals[owner[selected]] <= k
    pred = A.copy(); pred[selected] = E[selected]
    result_owner = np.where(selected, owner, -1).astype(np.int32)
    np.testing.assert_array_equal(pred[A != 0], A[A != 0])
    return pred, result_owner


def quality_from_transfer(transfer, ignored=0, accepted=0):
    t = np.asarray(transfer, np.int64)
    if t.shape != (21, 21) or (t < 0).any() or t[:, 0].any():
        raise ValueError('Extra transfers must be nonnegative and add foreground only')
    correct = int(np.trace(t)); bg = int(t[0, 1:].sum())
    wrong = int(t[1:, 1:].sum() - correct); valid = int(t.sum()); total = valid + int(ignored)
    return dict(added_pixels=total, valid_pixels=valid, correct_foreground_pixels=correct,
                background_false_positive_pixels=bg, wrong_foreground_class_pixels=wrong,
                ignore_pixels=int(ignored), accepted_candidate_count=int(accepted),
                precision_all=correct/total if total else None, precision_valid=correct/valid if valid else None,
                correct_per_added_pixel=correct/total if total else None,
                correct_per_accepted_candidate=correct/accepted if accepted else None,
                FP_per_recovered_FG=bg/correct if correct else None,
                net_valid_benefit=correct-bg-wrong)


def quality_group(correct, valid):
    if not valid: return 'UNAVAILABLE_NO_VALID_PIXELS'
    ratio = correct/valid
    return 'GOOD' if ratio >= .8 else 'BAD' if ratio < .5 else 'MID'


def confidence_diagnostic(original, crop, i):
    so, sc = float(original['retrieval_scores'][i]), float(crop['retrieval_scores'][i])
    mo, mc = float(original['margins'][i]), float(crop['margins'][i])
    a = np.asarray(original['class_scores'][i], np.float64); b = np.asarray(crop['class_scores'][i], np.float64)
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return dict(score_original=so, score_crop=sc, score_min=min(so, sc), abs_score_difference=abs(so-sc),
                margin_original=mo, margin_crop=mc, margin_min=min(mo, mc),
                class_scores_cosine=float(a@b/denom) if denom else None)
