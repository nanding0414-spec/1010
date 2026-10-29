"""Fixed cross-view R3 policies. No files, GT, training or parameter search."""
from __future__ import annotations
import numpy as np

PROTOCOL = 'r3_crossview_agreement_v1'
SEEDS = (42, 123, 2026)
CASES = ('A', 'B', 'C', 'D')
RULES = dict(crop_fraction=.8, resolution=1008, rho=.1, precision='fp32',
             centroid_weight=.2, topk=25, class_topm=5, search_multiplier=2,
             proposal=.4, strict_score=.95, strict_margin=.1, loose_score=.90,
             loose_margin=.05, novel_fraction=.1, proposal_weight=.2,
             shuffle_seed=42, shuffle_condition='mean_D_strictly_greater_than_mean_A',
             C_order='original_view_fused_score_then_candidate_id',
             D_order='A_first_then_min_view_score_fused_then_candidate_id',
             novel='live_background_after_A_and_previous_extras', new_training=0)


def validate_views(original, crop):
    n = len(original['class_ids'])
    for q in (original, crop):
        for k in ('class_ids', 'retrieval_scores', 'margins'):
            if np.asarray(q[k]).shape != (n,):
                raise ValueError('View score/label rows differ')
        s = np.asarray(q['class_scores'])
        labels = np.asarray(q['class_ids'])
        if (s.shape != (n, 21) or s.dtype != np.float32 or not np.isfinite(s).all()
                or not np.isin(labels, np.arange(21)).all()
                or not np.isfinite(q['retrieval_scores']).all() or not np.isfinite(q['margins']).all()):
            raise ValueError('Expected finite FP32 21-class scores in VOC ID order')
        if n:
            top = np.sort(s, axis=1)[:, -2:]
            np.testing.assert_allclose(s[np.arange(n), labels], q['retrieval_scores'], atol=2e-7, rtol=0)
            np.testing.assert_allclose(top[:, 1], q['retrieval_scores'], atol=2e-7, rtol=0)
            np.testing.assert_allclose(top[:, 1]-top[:, 0], q['margins'], atol=2e-7, rtol=0)


def crop_geometry(height, width):
    from tools.oracle_guided_r import center_crop_box
    return center_crop_box(height, width)


def assert_geometry(crop_masks, mapped_masks, box, original_hw):
    x0, y0, x1, y1 = map(int, box)
    if crop_masks.shape != (len(mapped_masks), y1-y0, x1-x0):
        raise ValueError('Use native crop masks with crop tokens, not original mask coordinates')
    if mapped_masks.shape != (len(crop_masks), *original_hw):
        raise ValueError('Original mask dimensions differ')
    back = np.zeros_like(mapped_masks)
    back[:, y0:y1, x0:x1] = crop_masks
    if not np.array_equal(back, mapped_masks):
        raise ValueError('Native crop and mapped mask identities differ')


def query_fields(view, proposal_scores, candidate_ids, source_classes):
    out = dict(view, proposal_scores=np.asarray(proposal_scores), candidate_ids=np.asarray(candidate_ids),
               source_classes=np.asarray(source_classes))
    out['fused_scores'] = .8 * np.asarray(view['retrieval_scores']) + .2 * np.asarray(proposal_scores)
    return out


def subset(q, keep):
    return {k: np.asarray(v)[keep] for k, v in q.items()}


def strict_owners(base, masks, ids, report):
    owner = np.full(base.shape, -1, np.int32)
    occupied = base != 0
    index = {str(v): i for i, v in enumerate(ids)}
    for row in report['accepted_candidates']:
        i = index[row['candidate_id']]
        added = masks[i].astype(bool) & ~occupied
        if int(added.sum()) != row['added_pixels']:
            raise ValueError('Strict candidate ownership differs from the production replay')
        owner[added] = i
        occupied |= added
    return owner


def extras(current_A, masks, original, crop, proposal_scores, candidate_ids, accepted_A):
    """A pixels are immutable; novel area is measured at each actual decision."""
    result = current_A.copy()
    ids = np.asarray(candidate_ids)
    ro, rc = np.asarray(original['retrieval_scores']), np.asarray(crop['retrieval_scores'])
    mo, mc = np.asarray(original['margins']), np.asarray(crop['margins'])
    lo, lc = np.asarray(original['class_ids']), np.asarray(crop['class_ids'])
    pp = np.asarray(proposal_scores)
    agreed = (lo == lc)
    # Match production's Python-float comparisons to the saved FP32 values.
    fp64 = lambda x: np.asarray(x, dtype=np.float64)
    eligible = (agreed & (lo > 0) & (lo < 21) & (fp64(pp) >= .4)
                & (fp64(ro) >= .9) & (fp64(rc) >= .9) & (fp64(mo) >= .05) & (fp64(mc) >= .05)
                & ~np.isin(ids, list(accepted_A)))
    rank = .8 * np.minimum(ro, rc) + .2 * pp
    order = sorted(np.flatnonzero(eligible), key=lambda i: (-float(rank[i]), str(ids[i])))
    owner = np.full(result.shape, -1, np.int32)
    traces = []
    for i in range(len(ids)):
        area = int(masks[i].sum())
        traces.append(dict(index=i, candidate_id=str(ids[i]), eligible=bool(eligible[i]),
                           accepted=False, added_pixels=0, novel_fraction=None,
                           novel_fraction_after_A=float((masks[i] & (current_A == 0)).sum()/max(area, 1))))
    for i in order:
        new = masks[i].astype(bool) & (result == 0)
        fraction = int(new.sum()) / max(int(masks[i].sum()), 1)
        traces[i]['novel_fraction'] = fraction
        if new.any() and fraction >= .1:
            result[new] = int(lo[i])
            owner[new] = int(i)
            traces[i].update(accepted=True, added_pixels=int(new.sum()))
    np.testing.assert_array_equal(result[current_A != 0], current_A[current_A != 0])
    return result, owner, traces


def compose(base, masks, proposal_scores, candidate_ids, source_classes, original, crop):
    from tools.oracle_guided_r import strict_background_rescue
    validate_views(original, crop)
    qo = query_fields(original, proposal_scores, candidate_ids, source_classes)
    qc = query_fields(crop, proposal_scores, candidate_ids, source_classes)
    A, ra = strict_background_rescue(base, masks, qo)
    B, rb = strict_background_rescue(base, masks, qc)
    keep = ((original['class_ids'] == crop['class_ids']) & (crop['retrieval_scores'].astype(np.float64) >= .95)
            & (crop['margins'].astype(np.float64) >= .1))
    C, cr = strict_background_rescue(base, masks[keep], subset(qo, keep))
    D, od, traces = extras(A, masks, original, crop, proposal_scores, candidate_ids,
                         {r['candidate_id'] for r in ra['accepted_candidates']})
    # C indices are mapped back to the common candidate pool, not its filtered rows.
    owners = {k: strict_owners(base, masks, candidate_ids, report)
              for k, report in [('A', ra), ('B', rb), ('C', cr)]}
    owners['D_extra'] = od
    return dict(A=A, B=B, C=C, D=D), owners, dict(A=ra, B=rb, C=cr, D_extra=traces)


def diagnostics(qo, qc):
    validate_views(qo, qc)
    agree = np.asarray(qo['class_ids']) == np.asarray(qc['class_ids'])
    score_edges = [-np.inf, .8, .85, .9, .95, np.inf]
    margin_edges = [0, .02, .05, .10, np.inf]
    def bins(values, edges):
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (values >= low) & (values < high)
            n = int(mask.sum()); a = int(agree[mask].sum())
            rows.append(dict(lower=None if not np.isfinite(low) else low,
                             upper=None if not np.isfinite(high) else high,
                             count=n, agreement_count=a, agreement_rate=a/n if n else None))
        return rows
    classes = []
    for cid in range(21):
        mask = np.asarray(qo['class_ids']) == cid
        n = int(mask.sum())
        ro, rc = qo['retrieval_scores'][mask], qc['retrieval_scores'][mask]
        classes.append(dict(class_id=cid, count=n, agreement_count=int(agree[mask].sum()),
                            agreement_rate=float(agree[mask].mean()) if n else None,
                            mean_score_o=float(ro.mean()) if n else None,
                            mean_score_c=float(rc.mean()) if n else None,
                            mean_abs_score_difference=float(np.abs(ro-rc).mean()) if n else None))
    return dict(score_bins=bins(qo['retrieval_scores'].astype(np.float64), score_edges),
                margin_bins=bins(qo['margins'].astype(np.float64), margin_edges), classes=classes,
                margin_below_zero=int((qo['margins'] < 0).sum()), no_GT_read=True)


def shuffle_indices(candidate_ids):
    ids = list(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Shared candidate identities must be unique')
    return np.random.default_rng(42).permutation(len(ids))


def fuse_captured_maps(trunk, fpn):
    """Exact original feature-extractor operations, using this forward's maps."""
    import torch
    from torch.nn import functional as F
    if trunk.shape != (1, 1024, 72, 72) or fpn.shape != (1, 256, 72, 72):
        raise ValueError('Expected native stride14 trunk1024 + FPN256 maps')
    if trunk.dtype != torch.float32 or fpn.dtype != torch.float32:
        raise ValueError('Cross-view features must stay FP32')
    def normalize(x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2).contiguous()
        return F.normalize(x, p=2, dim=1)
    result = torch.cat([normalize(trunk), normalize(fpn)], dim=1)
    return result.permute(0, 2, 3, 1).reshape(5184, 1280).float()


class Capture:
    """Read-only hooks: exactly one real trunk/FPN call, no extra encoder."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.maps = {}
        self.counts = dict(trunk=0, fpn=0, input=0)

    def __enter__(self):
        neck = self.model.backbone.vision_backbone
        def trunk(_module, _args, output):
            self.counts['trunk'] += 1
            x = output[-1]
            self.maps['trunk'] = (x.tensors if hasattr(x, 'tensors') else x).detach().clone()
        def fpn(_module, _args, output):
            self.counts['fpn'] += 1
            self.maps['fpn'] = output.detach().clone()
        def inp(_module, args):
            self.counts['input'] += 1
            self.maps['input'] = args[0].detach().clone()
        self.handles = [neck.trunk.register_forward_pre_hook(inp), neck.trunk.register_forward_hook(trunk),
                        neck.convs[2].register_forward_hook(fpn)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def tokens(self):
        if self.counts != dict(trunk=1, fpn=1, input=1):
            raise ValueError('Crop proposals/tokens require exactly one corresponding forward')
        return fuse_captured_maps(self.maps['trunk'], self.maps['fpn'])
