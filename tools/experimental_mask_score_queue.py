#!/usr/bin/env python3
"""Fixed confidence-weighted CE x proposal-score fusion experiment (worker isolated)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_gmp_mlp2 as previous
mlp, queue, ab, gmp, mpsa, seeds = (previous.mlp, previous.queue, previous.ab,
                                  previous.gmp, previous.mpsa, previous.seeds)
SEEDS = (42, 123, 2026)
ALPHAS = (0., .1, .2, .4)
PROTOCOL = 'mask_score_ce_fusion_v1'
FORMULA = 'foreground=.5+.5*mean_pixel_max_same_class_score;background=1;CE=sum(w*ce)/sum(w)'
ACTIVE_KEYS = ('bucket_size', 'preview', 'preview_resolution', 'presence_threshold',
               'active_keep_ratio', 'min_active_classes', 'max_active_classes', 'neighbor_tokens',
               'hypernym_tokens', 'context_strength', 'min_mask_region_area', 'max_mask_area_ratio',
               'mask_nms_threshold')


class ScoreReplayFailure(queue.TaskFailure):
    """Only the confidence training branch is blocked."""


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cell(mode, seed, alpha):
    if mode not in ('original', 'weighted') or seed not in SEEDS or alpha not in ALPHAS:
        raise ValueError('Outside the fixed experiment matrix')
    return f'{mode}_{seed}_a{round(alpha * 10)}'


def region_scores(old_seg, masks, scores, class_ids, allowed, regions):
    """Reproduce old class-first union and score its original connected regions."""
    import numpy as np
    old_seg = np.asarray(old_seg)
    masks, scores, class_ids = np.asarray(masks), np.asarray(scores), np.asarray(class_ids)
    if (old_seg.ndim != 2 or masks.shape != (len(scores), *old_seg.shape)
            or class_ids.shape != scores.shape or scores.ndim != 1
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not np.isin(masks, [0, 1]).all() or not np.isin(class_ids, allowed).all()):
        raise ScoreReplayFailure('Invalid replay masks/classes/scores')
    reconstructed = np.zeros_like(old_seg)
    confidence = np.zeros(old_seg.shape, np.float32)
    covered = np.zeros(old_seg.shape, bool)
    for cls in dict.fromkeys(allowed):
        local = np.zeros(old_seg.shape, np.float32)
        union = np.zeros(old_seg.shape, bool)
        for mask, score in zip(masks[class_ids == cls], scores[class_ids == cls]):
            mask = mask.astype(bool)
            union |= mask
            local[mask] = np.maximum(local[mask], score)
        assigned = union & (reconstructed == 0)
        reconstructed[assigned] = cls
        confidence[assigned] = local[assigned]
        covered |= assigned
    difference = int(np.count_nonzero(reconstructed != old_seg))
    if difference:
        raise ScoreReplayFailure(f'Replayed pseudo PNG differs at {difference} pixels')
    rows = []
    for region in regions:
        mask = np.asarray(region['mask'], dtype=bool)
        cls = int(region['cls_id'])
        if mask.shape != old_seg.shape or not mask.any() or not np.all(old_seg[mask] == cls):
            raise ScoreReplayFailure('Region geometry/class differs')
        if cls and not covered[mask].all():
            raise ScoreReplayFailure('Foreground region has uncovered pixels')
        q = float(confidence[mask].mean(dtype=np.float64)) if cls else None
        rows.append(dict(score=q, weight=1. if q is None else .5 + .5 * q))
    return rows


def weighted_ce(logits, targets, weights, label_smoothing=0.):
    import torch
    from torch.nn import functional as F
    if (weights.shape != targets.shape or not torch.isfinite(weights).all()
            or (weights < .5).any() or (weights > 1).any()):
        raise ValueError('Invalid per-row confidence weights')
    # Keep the unweighted control bit-identical to the original reduction.
    if bool((weights == 1).all()):
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    losses = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    return (losses * weights).sum() / weights.sum()


def align_weights(records, score_rows):
    import numpy as np
    lookup = {}
    for row in score_rows:
        key = row['cls'], row['id']
        if key in lookup or not row.get('mask_sha256'):
            raise ValueError('Duplicate or unbound confidence row')
        q = row['score']
        w = row['weight']
        if row['cls'] == 'background':
            expected = 1.
            if q is not None:
                raise ValueError('Background must not have a fabricated SAM3 score')
        else:
            if q is None or not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError('Invalid foreground confidence')
            expected = .5 + .5 * q
        if not math.isfinite(w) or w != expected:
            raise ValueError('Confidence formula mismatch')
        lookup[key] = w
    ordered = [(relative.parent.name, str(sid)) for relative, features, ids in records for sid in ids]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(lookup):
        raise ValueError('Loaded bank rows and confidence members differ')
    return np.array([lookup[key] for key in ordered], dtype=np.float32)


def validate_eval(current, reference, ids, count, alpha):
    if current['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT'] != alpha:
        raise ValueError('Effective fusion alpha differs')
    ref = canonical(reference)
    ref['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT'] = alpha
    mlp.validate_metrics(current, ref, ids, count)


def require_image_replay(a, b):
    import numpy as np
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        if any(not np.array_equal(x[k], y[k]) for k in ('ids', 'confusions')):
            raise queue.GlobalFailure('Historical per-image confusion matrices differ')


def checked_history(folder, clean_dir):
    manifest = ab.read(folder / 'manifest.json')
    if manifest['protocol'] != previous.PROTOCOL or ab.read(folder / 'status.json')['status'] != 'complete':
        raise ValueError('Need a complete GMP + two-layer experiment')
    common, _, split, records, clean, raw, signature = previous.history(
        Path(manifest['combo_previous']), Path(manifest['mlp_previous']), clean_dir)
    for field in ('code', 'references'):
        gmp.verify(manifest[field], 'GMP2 ' + field)
        records += manifest[field]
    records += ab.identities([folder / n for n in ('manifest.json', 'status.json', 'summary.json')])
    refs = {}
    for seed in SEEDS:
        value = dict(raw=raw)
        for kind in ('train', 'eval'):
            marker = folder / f'{kind}_{seed}' / 'complete.json'
            value.update(queue.verified_stage(folder, f'{kind}_{seed}', records))
            records += ab.identities([marker])
        report = ab.read(Path(value['weights']) / 'training_report.json')
        if (report['args']['seed'] != seed or report['args']['num_layers'] != 2
                or Path(report['args']['source_bank']).resolve() != Path(raw).resolve()
                or any(report[k] != split[k] for k in split)):
            raise ValueError('Wrong GMP2 training recipe/split')
        sig = ab.read(Path(value['bank']) / 'meta/feature_signature.json')
        if sig['adapter']['architecture']['experimental_mlp'] != mlp.adapter_tag('d2'):
            raise ValueError('Wrong adapter architecture')
        underlying = dict(sig); underlying.pop('adapter'); underlying['embed_dim'] = 1280
        if underlying != signature or ab.bank_members(value['bank']) != ab.bank_members(raw):
            raise ValueError('Different GMP feature space/members')
        metric = ab.read(value['metrics'])
        validate_eval(metric, metric, common['val_ids'], clean['count'], .2)
        if metric['feature_signature'] != sig:
            raise ValueError('Historical evaluated model differs')
        seeds.validate_image_matrices(value['per_image'], metric)
        refs[seed] = value
    return common, refs, split, records, clean, raw, signature


def generation_recipe(common):
    """Metadata is mandatory; missing semantic settings never inherit new defaults."""
    files = sorted(Path(common['pseudo']).rglob('generation_*.json'))
    if not files:
        raise ScoreReplayFailure('No historical generation metadata beneath pseudo directory')
    values = [ab.read(p) for p in files]
    recipe = values[0]
    if any(v != recipe for v in values):
        raise ScoreReplayFailure('Conflicting generation metadata/shards')
    required = {'format_version', 'dataset', 'split', 'conf_thr', 'min_mask_area', 'use_bf16',
                'dataset_root', 'checkpoint', 'label_json_sha256', 'prompt_backend',
                'active_options', 'active_classes', 'prompts', 'image_dir', 'vit_mlp_precision_policy'}
    if not required <= recipe.keys() or not set(ACTIVE_KEYS) <= recipe['active_options'].keys():
        raise ScoreReplayFailure('Incomplete historical effective Active settings')
    if (recipe['format_version'] != 2 or recipe['dataset'] != 'voc' or recipe['prompt_backend'] != 'active'
            or recipe['use_bf16'] or recipe['vit_mlp_precision_policy'] != 'standard_non_bf16_v1'
            or recipe.get('output_size_reference_dir')):
        raise ScoreReplayFailure('Unsupported historical pseudo protocol (need original-image FP32)')
    for key in ('checkpoint', 'dataset_root', 'image_dir'):
        if not recipe[key] or Path(recipe[key]).resolve() != Path(common[key]).resolve():
            raise ScoreReplayFailure('Generation source mismatch: ' + key)
    if recipe['label_json_sha256'] != ab.digest(common['imagelevel_json']):
        raise ScoreReplayFailure('Training image-level labels changed')
    # max_masks was not exposed by the historical generator CLI; bind its verified
    # source default explicitly, instead of letting a future constructor choose it.
    recipe = canonical(recipe)
    recipe['active_options'].setdefault('max_masks', 0)
    return recipe, ab.identities(files)


def score_cache_identity(common, recipe, plan, signature, code):
    return dict(protocol=PROTOCOL, formula=FORMULA, provenance='replayed_not_historical_scores',
        recipe=recipe, source_code=code, checkpoint_sha256=common['checkpoint_sha256'],
        train_ids=common['train_ids'], plan_sha256=object_hash(plan), raw_signature=signature,
        inputs={i: common['cache_identity'][i] for i in common['train_ids']})


def read_score_image(root, stem, identity, expected):
    path = Path(root) / f'{stem}.json'
    if not path.exists():
        return None
    envelope = ab.read(path)
    payload = envelope['payload']
    if envelope['sha256'] != object_hash(payload) or payload['identity'] != identity or payload['image'] != stem:
        raise queue.GlobalFailure('Changed score cache: ' + stem)
    rows = payload['rows']
    if [{k: r[k] for k in ('index', 'cls', 'id', 'mask_sha256')} for r in rows] != expected:
        raise queue.GlobalFailure('Score member order/mask identity changed: ' + stem)
    return rows


def score_worker(job):
    import numpy as np
    from PIL import Image
    import torch
    ab.install_precision()
    from project_utils.seed_utils import set_global_seed
    set_global_seed(42)
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    from src.region_extractors import get_regions_from_pseudo
    r = job['recipe']
    generator = SAM3PseudoMaskGenerator(dataset_type='voc', ckpt=job['checkpoint'],
        conf_thr=r['conf_thr'], use_bf16=False, min_mask_area=r['min_mask_area'],
        prompt_backend='active', active_options=r['active_options'], dataset_root=job['dataset_root'],
        image_dir=job['image_dir'], imagelevel_json=job['imagelevel_json'], device='cuda:0',
        output_dir=job['score_root'])
    if canonical([asdict(c) for c in generator.active_classes]) != r['active_classes'] or generator.classes != r['prompts']:
        raise ScoreReplayFailure('Active vocabulary differs')
    _, labels = generator.get_image_list_and_labels(r['split'])
    proposer = generator.active_proposer
    captured = {}
    def capture(*args, **kwargs):
        out = proposer(*args, **kwargs)
        captured.update(masks=out['instances'].pred_masks.cpu().numpy(),
            scores=out['instances'].scores.cpu().numpy(), class_ids=proposer.last_result['class_ids'])
        return out
    class Capture:
        def __call__(self, *args, **kwargs): return capture(*args, **kwargs)
        def __getattr__(self, name): return getattr(proposer, name)
    generator.active_proposer = Capture()
    done = 0
    for stem in job['score_ids']:
        if read_score_image(job['score_root'], stem, job['score_identity_hash'], job['plan'][stem]) is not None:
            done += 1
            continue
        queue.ensure_resources(job['deadline'])
        start = time.monotonic()
        old_path = Path(job['pseudo']) / f'{stem}.png'
        if (ab.digest(old_path) != job['cache_identity'][stem]['mask_source_sha256']
                or ab.digest(job['images'][stem]) != job['cache_identity'][stem]['image_sha256']):
            raise queue.GlobalFailure('Training input changed: ' + stem)
        image_path = Path(job['images'][stem])
        image_labels = labels.get(str(image_path), labels.get(stem))
        if image_labels is None: raise ScoreReplayFailure('Missing image-level labels: ' + stem)
        allowed = list(dict.fromkeys(int(x) for x in image_labels if int(x) != 0))
        generated = generator.generate_mask(image_path, allowed)
        with Image.open(old_path) as im: old = np.asarray(im)
        differences = int(np.count_nonzero(generated != old)) if generated.shape == old.shape else -1
        if differences:
            ab.write(Path(job['attempt']) / 'mismatch.json', dict(image=stem, differing_pixels=differences,
                expected_shape=list(old.shape), replay_shape=list(generated.shape)))
            raise ScoreReplayFailure(f'PNG replay mismatch: {stem}, pixels={differences}')
        regions = get_regions_from_pseudo(image_path, 300, job['pseudo'])
        values = region_scores(old, captured['masks'], captured['scores'], captured['class_ids'], allowed, regions)
        rows = []
        for row in job['plan'][stem]:
            index = row['index']
            if ab_hash_mask(regions[index]['mask']) != row['mask_sha256']:
                raise queue.GlobalFailure('Training region changed')
            rows.append(dict(row, **values[index]))
        payload = dict(identity=job['score_identity_hash'], image=stem, rows=rows,
            seconds=time.monotonic()-start, candidate_count=len(captured['scores']), png_equal=True)
        path = Path(job['score_root']) / f'{stem}.json'
        if path.exists(): raise queue.GlobalFailure('Refusing to overwrite score sidecar')
        ab.write(path, dict(payload=payload, sha256=object_hash(payload)))
        done += 1
        print(f'[Scores] {done}/{len(job["score_ids"])} {stem}', flush=True)
    return dict(images=done)


def ab_hash_mask(mask):
    return hashlib.sha256(mask.tobytes()).hexdigest()


def install_confidence_adapter(tag):
    """Bind confidence training provenance to the adapter, not the raw space."""
    import src.feature_adapter as fa
    factory, loader, signature = fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature
    def create(*args, score_training=None, **kwargs):
        if score_training is not None and score_training != tag:
            raise ValueError('Confidence training identity mismatch')
        model = factory(*args, **kwargs)
        model.architecture['score_training'] = tag
        return model
    def load(path, device='cpu', source_signature=None):
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('architecture', {}).get('score_training') != tag:
            raise ValueError('Expected matching confidence-weighted checkpoint')
        return loader(path, device, source_signature)
    def adapted(source, checkpoint, architecture):
        if architecture.get('score_training') != tag: raise ValueError('Wrong score provenance')
        value = signature(source, checkpoint, architecture)
        value['adapter']['type'] = PROTOCOL
        return value
    fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature = create, load, adapted


def validate_job(job):
    if job['stage'] == 'scores':
        if not job.get('recipe') or job['score_identity_hash'] != job['score_tag']['cache_identity']:
            raise ValueError('Missing score replay identity')
        return
    if job['stage'] not in ('train', 'eval') or job.get('group') != 'C':
        raise ValueError('Only fixed GMP C train/eval is permitted')
    cell(job['training_mode'], job['seed'], job['alpha'])
    if job['stage'] == 'train' and job['training_mode'] != 'weighted':
        raise ValueError('Historical baseline is read-only')
    expected = queue.signature({}, 'combo', queue.Recipe('C', job['checkpoint_sha256']))
    if any(job['raw_signature'][k] != expected[k] for k in ('format_version','experimental_pooling')):
        raise ValueError('Aggregation/phase/rho/base weight mismatch')
    if job['training_mode'] == 'weighted':
        tag = job['score_tag']
        if (tag['protocol'] != PROTOCOL or tag['formula'] != FORMULA or len(tag['cache_identity']) != 64
                or len(tag.get('rows_sha256', '')) != 64):
            raise ValueError('Confidence recipe mismatch')


def worker(job):
    validate_job(job)
    os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'sam3-main'))
    import torch
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); status, category, error = 'failed', 'local', None
    try:
        if job['stage'] == 'scores':
            score_worker(job)
        else:
            if job['stage'] not in ('train', 'eval'): raise ValueError('Unsupported stage')
            cell(job['training_mode'], job['seed'], job['alpha'])
            if job['stage'] == 'train' and job['training_mode'] != 'weighted':
                raise ValueError('Historical baseline is read-only')
            import configs.config as cfg
            cfg.INF_PROPOSAL_SCORE_WEIGHT = job['alpha']
            mlp.install_adapter('d2')
            if job['training_mode'] == 'weighted': install_confidence_adapter(job['score_tag'])
            def run(path, *args, **kwargs):
                if Path(path).resolve() != ROOT / 'tools/train_feature_adapter.py':
                    raise ValueError('Unexpected training entry')
                sys.argv = previous.depth_two_argv(sys.argv)
                ab.write(Path(job['attempt']) / 'effective_training_argv.json', dict(argv=sys.argv,
                    seed=job['seed'], architecture=mlp.adapter_tag('d2'), score_training=job['score_tag']))
                return train_weighted(job)
            ab.runpy.run_path = run
            queue.worker(job)
        torch.cuda.synchronize(); status = 'complete'
    except queue.Deferred as exc:
        category, error = 'deferred', str(exc); raise
    except ScoreReplayFailure as exc:
        category, error = 'score_branch', str(exc); raise
    except torch.cuda.OutOfMemoryError as exc:
        category, error = 'oom', str(exc); raise
    except (ValueError, KeyError, FileNotFoundError, queue.GlobalFailure) as exc:
        category, error = 'global', str(exc); raise
    except BaseException as exc:
        error = str(exc); raise
    finally:
        ab.write(Path(job['attempt']) / 'queue_worker.json', dict(status=status, category=category,
            error=error, seconds=time.monotonic()-started, stage=job['stage'], precision_policy=ab.POLICY,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def summary(reports):
    groups, contrasts = {}, {}
    for mode in ('original', 'weighted'):
        for alpha in ALPHAS:
            values = {s: reports[cell(mode, s, alpha)]['mIoU'] for s in SEEDS if cell(mode, s, alpha) in reports}
            complete = len(values) == 3
            groups[f'{mode}_a{round(alpha*10)}'] = dict(complete=complete, by_seed=values,
                mean_miou=statistics.mean(values.values()) if complete else None,
                sample_std_miou=statistics.stdev(values.values()) if complete else None)
    formulas = {'training': [('weighted', .2, 1), ('original', .2, -1)],
                'proposal_score': [('original', .2, 1), ('original', 0., -1)]}
    for mode in ('original', 'weighted'):
        for alpha in (.1, .4):
            formulas[f'{mode}_a{round(alpha*10)}-a2'] = [(mode, alpha, 1), (mode, .2, -1)]
    for alpha in (0., .1, .4):
        formulas[f'interaction_a{round(alpha*10)}'] = [('weighted', alpha, 1), ('original', alpha, -1),
                                                     ('weighted', .2, -1), ('original', .2, 1)]
    for key, terms in formulas.items():
        values = {s: 100*sum(sign*reports[cell(mode,s,a)]['mIoU'] for mode,a,sign in terms)
                  for s in SEEDS if all(cell(mode,s,a) in reports for mode,a,_ in terms)}
        contrasts[key] = dict(complete=len(values)==3, by_seed_pp=values,
            mean_pp=statistics.mean(values.values()) if len(values)==3 else None,
            sample_std_pp=statistics.stdev(values.values()) if len(values)==3 else None)
    return dict(groups=groups, contrasts=contrasts)


def train_weighted(job):
    """Snapshot of v1 trainer: identical sampling/selection/export, weighted CE only."""
    import random
    import shutil
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    from src.adapter_training import (load_training_bank, split_by_image, balanced_batch,
                                      retrieval_metrics, split_from_report)
    from src.feature_adapter import (FeatureAdapter, supervised_contrastive_loss,
                                     project_features, load_adapter, adapted_signature,
                                     CosineClassifier, feature_consistency_loss)
    from src.feature_bank_storage import save_feature_matrix_npz
    from src.feature_signature import save_feature_signature
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--train-list", required=True, help="Allowed TRAIN image stems, one per line")
    parser.add_argument("--output-dir", required=True, help="New directory for weights and metrics")
    parser.add_argument("--export-bank", required=True, help="New projected bank directory")
    parser.add_argument("--recipe", choices=["v1", "v2"], default="v1",
                        help="v2: cosine classifier, regularization, balanced checkpoint selection")
    parser.add_argument("--classifier", choices=["linear", "cosine"], default=None)
    parser.add_argument("--selection-metric", choices=["knn", "balanced"], default=None)
    parser.add_argument("--feature-dropout", type=float, default=None)
    parser.add_argument("--consistency-weight", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--cosine-scale", type=float, default=16.0)
    parser.add_argument("--eval-class-topm", type=int, default=5)
    parser.add_argument("--eval-centroid-weight", type=float, default=0.4)
    parser.add_argument("--eval-search-multiplier", type=int, default=2)
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--instances-per-class", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, choices=[2, 3], default=3,
                        help="Number of Linear layers in the adapter (default: 3)")
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path,
                        help="Reuse validation image membership from a baseline training_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    from configs.experiment import TRAINING
    parser.set_defaults(**TRAINING)
    args = parser.parse_args()
    defaults = (dict(classifier="cosine", selection_metric="balanced", feature_dropout=0.1,
                     consistency_weight=0.2, label_smoothing=0.05, eval_topk=25)
                if args.recipe == "v2" else
                dict(classifier="linear", selection_metric="knn", feature_dropout=0.0,
                     consistency_weight=0.0, label_smoothing=0.0, eval_topk=5))
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.epochs, args.steps_per_epoch, args.hidden_dim, args.output_dim) < 1:
        parser.error("epochs, steps and dimensions must be positive")
    if args.classes_per_batch < 2 or args.instances_per_class < 2:
        parser.error("Use at least two classes and two instances per class")
    if args.lr <= 0 or args.temperature <= 0 or args.contrastive_weight < 0:
        parser.error("Invalid learning rate, temperature or contrastive weight")
    if not 0 <= args.feature_dropout < 1 or not 0 <= args.label_smoothing < 1:
        parser.error("feature-dropout and label-smoothing must be in [0, 1)")
    if args.consistency_weight < 0 or args.cosine_scale <= 0:
        parser.error("Invalid consistency weight or cosine scale")
    if min(args.eval_class_topm, args.eval_search_multiplier, args.eval_topk) < 1:
        parser.error("Evaluation neighbor counts must be positive")
    if not 0 <= args.eval_centroid_weight <= 1:
        parser.error("eval-centroid-weight must be in [0, 1]")
    out = Path(args.output_dir).resolve()
    bank_out = Path(args.export_bank).resolve()
    source = Path(args.source_bank).resolve()
    for target in (out, bank_out):
        if target.exists() or target == source or source in target.parents or target in source.parents:
            parser.error(f"Choose a fresh directory outside the source bank: {target}")
    if out == bank_out or out in bank_out.parents or bank_out in out.parents:
        parser.error("Weights and exported bank must be separate directories")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y, groups, classes, signature, records = load_training_bank(source, args.train_list)
    if args.split_file:
        args.split_file = str(args.split_file.resolve())
    train, val = (split_from_report(groups, y, args.split_file) if args.split_file
                  else split_by_image(groups, y, args.val_fraction, args.seed))
    bg_index = classes.index("background") if "background" in classes else None
    if not any(len(np.unique(groups[train[y[train] == cls]])) >= 2
               for cls in np.unique(y) if cls != bg_index):
        raise ValueError("No foreground class has two distinct training images for positive pairs")
    confidence = ab.read(job['score_rows'])
    if ab.digest(job['score_rows']) != job['score_rows_sha256']:
        raise ValueError('Confidence table changed before training')
    if job['score_rows_sha256'] != job['score_tag']['rows_sha256']:
        raise ValueError('Adapter tag does not bind the actual confidence table')
    row_weights = align_weights(records, confidence['rows'])
    if confidence['identity'] != job['score_tag']['cache_identity']:
        raise ValueError('Confidence source identity mismatch')
    if len(row_weights) != len(x):
        raise ValueError('Confidence rows differ from loaded features')
    out.mkdir(parents=True)
    device = torch.device(args.device)
    model = FeatureAdapter(x.shape[1], args.hidden_dim, args.output_dim,
                           num_layers=args.num_layers).to(device)
    print(f"[Adapter] {model.architecture}")
    classifier = (CosineClassifier(args.output_dim, len(classes), args.cosine_scale)
                  if args.classifier == "cosine" else nn.Linear(args.output_dim, len(classes))).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    baseline = retrieval_metrics(x[train], y[train], x[val], y[val])
    evaluation = dict(k=args.eval_topk, mode=args.selection_metric,
                      class_topm=args.eval_class_topm, centroid_weight=args.eval_centroid_weight,
                      search_multiplier=args.eval_search_multiplier)
    selection_baseline = retrieval_metrics(x[train], y[train], x[val], y[val], **evaluation)
    print(f"[Data] {len(train)} train / {len(val)} held-out instances; {len(classes)} classes")
    print(f"[Raw SAM3 kNN] {baseline}")
    print(f"[Recipe] {args.recipe}, classifier={args.classifier}, selection={evaluation}")
    print(f"[Raw SAM3 selection baseline] {selection_baseline}")
    print("[Validation] Held-out TRAIN images, pseudo labels; these scores are not GT mIoU.")
    report = {"args": vars(args), "classes": classes, "baseline": baseline,
              "selection_baseline": selection_baseline,
              "train_images": sorted(set(groups[train].tolist())),
              "validation_images": sorted(set(groups[val].tolist())), "epochs": []}
    report['score_training'] = job['score_tag']
    report['confidence_by_class'] = {}
    for index, cls in enumerate(classes):
        weights = row_weights[train[y[train] == index]]
        report['confidence_by_class'][cls] = dict(count=len(weights), mean_weight=float(weights.mean()),
            weight_quantiles=np.quantile(weights, [0, .25, .5, .75, 1]).tolist(),
            mean_score=None if cls == 'background' else float((2*weights-1).mean()))
    report['sampled_weight_mass'] = {cls: 0. for cls in classes}
    report['normalized_ce_mass'] = {cls: 0. for cls in classes}
    report['sampled_count'] = {cls: 0 for cls in classes}
    checkpoint = out / "adapter.pt"
    best = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()
        losses = []
        components = []
        for _ in range(args.steps_per_epoch):
            rows = balanced_batch(y, groups, train, rng, args.classes_per_batch,
                                  args.instances_per_class)
            for index in np.unique(y[rows]):
                selected = rows[y[rows] == index]
                report['sampled_weight_mass'][classes[index]] += float(row_weights[selected].sum())
                report['normalized_ce_mass'][classes[index]] += float(row_weights[selected].sum()/row_weights[rows].sum())
                report['sampled_count'][classes[index]] += len(selected)
            batch_x = torch.from_numpy(x[rows]).to(device)
            batch_y = torch.from_numpy(y[rows]).long().to(device)
            z = model(batch_x)
            batch_weights = torch.from_numpy(row_weights[rows]).to(device)
            ce = weighted_ce(classifier(z), batch_y, batch_weights, args.label_smoothing)
            contrastive = supervised_contrastive_loss(z, batch_y, args.temperature, bg_index)
            consistency = (feature_consistency_loss(model, batch_x, z, args.feature_dropout)
                           if args.consistency_weight > 0 else z.sum() * 0)
            loss = ce + args.contrastive_weight * contrastive + args.consistency_weight * consistency
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite adapter training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append([float(ce.detach()), float(contrastive.detach()), float(consistency.detach())])
        scheduler.step()
        model.eval()
        z = project_features(model, x, device)
        metrics = retrieval_metrics(z[train], y[train], z[val], y[val], **evaluation)
        mean_components = np.mean(components, axis=0)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics,
               "ce_loss": float(mean_components[0]), "contrastive_loss": float(mean_components[1]),
               "consistency_loss": float(mean_components[2])}
        report["epochs"].append(row)
        print(f"[Epoch {epoch:03d}] loss={row['loss']:.4f} {args.selection_metric} macro={metrics['macro_accuracy']:.4f}")
        if metrics["macro_accuracy"] > best:
            best, best_epoch = metrics["macro_accuracy"], epoch
            payload = {"format_version": 1, "architecture": model.architecture,
                       "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "source_signature": signature, "classes": classes,
                       "epoch": epoch, "validation": metrics, "train_args": vars(args)}
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint)
    report.update(best_epoch=best_epoch, best_macro_accuracy=best)
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best <= selection_baseline["macro_accuracy"]:
        print("[Warning] Adapter did not beat raw-feature held-out retrieval. Keep the raw baseline for comparison.")

    model, _ = load_adapter(checkpoint, device, signature)
    bank_out.mkdir(parents=True)
    (bank_out / "meta").mkdir()
    shutil.copy2(source / "meta" / "class_mapping.json", bank_out / "meta" / "class_mapping.json")
    stats = {}
    for relative_path, features, ids in records:
        projected = project_features(model, features, device)
        save_feature_matrix_npz(bank_out / relative_path, projected, ids)
        stats[relative_path.parent.name] = {"count": len(projected)}
    save_feature_signature(bank_out / "meta", adapted_signature(signature, checkpoint, model.architecture))
    (bank_out / "meta" / "adapter_export.json").write_text(json.dumps({
        "source_bank": str(source), "adapter_checkpoint": str(checkpoint),
        "best_epoch": best_epoch, "counts": stats,
        "note": "Includes all source TRAIN instances after internal validation; no additional filtering."
    }, indent=2), encoding="utf-8")
    print(f"[Done] Adapter: {checkpoint}")
    print(f"[Done] Projected bank: {bank_out}; inference builds its FAISS index from these matrices.")


def verify_score_cache(root, identity, plan, ids, require_complete=True):
    root = Path(root)
    if ab.read(root / 'identity.json') != identity:
        raise queue.GlobalFailure('Score cache belongs to different inputs/code/recipe')
    rows = []
    for stem in ids:
        value = read_score_image(root, stem, object_hash(identity), plan[stem])
        if value is None:
            if require_complete: raise queue.GlobalFailure('Incomplete score cache: ' + stem)
        else:
            rows.extend(value)
    if require_complete:
        marker = ab.read(root / 'complete.json')
        if marker['identity'] != object_hash(identity): raise queue.GlobalFailure('Score manifest mismatch')
        gmp.verify(marker['artifacts'], 'Complete score cache')
        if {r['path'] for r in marker['artifacts']} != {str((root / f'{i}.json').resolve()) for i in ids}:
            raise queue.GlobalFailure('Score manifest coverage mismatch')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--previous', type=Path, default=Path('experiments/gmp_mlp2_20261002_141439'))
    parser.add_argument('--version', default='mask_score_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--max-hours', type=float, default=48)
    parser.add_argument('--clean-proposals-dir', type=Path, default=Path(mpsa.DEFAULT_CLEAN))
    parser.add_argument('--train-score-cache', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    opts = parser.parse_args(argv)
    os.chdir(ROOT)
    if opts.worker: return worker(ab.read(opts.worker))
    ab.parse_args(['--version', opts.version])
    if len(set(opts.gpus)) != 2 or min(opts.gpus) < 0 or not math.isfinite(opts.max_hours) or opts.max_hours <= 0:
        parser.error('Need two distinct GPUs and positive finite hours')
    for gpu in opts.gpus: ab.check_gpu(gpu)
    common, refs, split, records, clean, raw, signature = checked_history(opts.previous.resolve(), opts.clean_proposals_dir.resolve())
    # Original combo membership maps global pre-filter counters to region masks.
    source = Path(ab.read(opts.previous / 'manifest.json')['combo_previous'])
    queue.verified_stage(source, 'membership', records)
    records += ab.identities([source / 'membership/complete.json'])
    plan = ab.read(source / 'members.json')
    if set(plan) != set(common['train_ids']): raise ValueError('Membership image coverage changed')
    member_pairs = [(r['cls'], r['id']) for rows in plan.values() for r in rows]
    actual_pairs = {(c, sid) for c, ids in ab.bank_members(raw).items() for sid in ids}
    if len(member_pairs) != len(set(member_pairs)) or set(member_pairs) != actual_pairs:
        raise ValueError('Score membership plan differs from fixed GMP raw bank')
    import numpy as np
    for stem in common['val_ids']:
        with np.load(Path(common['proposals']) / f'{stem}.npz', allow_pickle=False) as data:
            scores = data['scores']
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError('Invalid proposal confidence')
    recipe, metadata_error = None, None
    try:
        recipe, generation_records = generation_recipe(common)
        records += generation_records
    except ScoreReplayFailure as exc:
        metadata_error = str(exc)
    code = ab.source_identity()
    score_identity = score_cache_identity(common, recipe, plan, signature, code)
    score_tag = dict(protocol=PROTOCOL, formula=FORMULA, cache_identity=object_hash(score_identity))
    output, logs = ROOT / 'experiments' / opts.version, ROOT / 'logs' / opts.version
    weights, banks = ROOT / 'weights/sam3_adapter' / opts.version, ROOT / 'storage/mlp_banks/voc' / opts.version
    own_scores = ROOT / 'storage/pseudo_labels/voc' / opts.version / 'score_sidecars'
    score_root = opts.train_score_cache.resolve() if opts.train_score_cache else own_scores
    if opts.train_score_cache:
        if metadata_error: raise ValueError(metadata_error)
        verify_score_cache(score_root, score_identity, plan, common['train_ids'])
        records += ab.identities(ab.files_under([score_root]))
    identity = canonical(dict(protocol=PROTOCOL, previous=str(opts.previous.resolve()), code=code,
        references=records, alphas=ALPHAS, seeds=SEEDS, gpus=opts.gpus,
        raw_signature=signature, score_identity=score_identity, metadata_error=metadata_error,
        external_scores=str(score_root) if opts.train_score_cache else None))
    owned = [output, logs, weights, banks] + ([] if opts.train_score_cache else [own_scores.parent])
    if output.exists():
        if not opts.resume or ab.read(output / 'manifest.json') != identity:
            raise ValueError('Version differs or --resume missing; use a new version for changed code/inputs')
    elif any(p.exists() for p in owned[1:]):
        raise FileExistsError('Output version already used')
    stages = ab.Stages(output, True); stages.verify_all()
    estimate = len(member_pairs)*256*4*3 + len(member_pairs)*2048 + 2**30
    if queue.shutil.disk_usage(ROOT).free < queue.FLOOR + (0 if opts.resume else estimate):
        raise queue.Deferred('Need space for 3 banks + score sidecars + 1 GiB overhead + 5 GiB reserve')
    print('[Plan] GMP2 x confidence CE x alpha 0/.1/.2/.4; 3 trainings, 24 full evaluations.', flush=True)
    print('[Score branch] ' + (metadata_error or 'Historical metadata checked; real PNG replay still required.'), flush=True)
    if opts.preflight_only:
        print('[Preflight passed] No workers started. Score branch may be blocked as reported above.', flush=True)
        return
    if not output.exists(): ab.write(output / 'manifest.json', identity)
    lock = output / 'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    budget = output / 'budget.json'
    if not budget.exists(): ab.write(budget, dict(started_at=time.time()))
    deadline = ab.read(budget)['started_at'] + opts.max_hours * 3600
    cancel, mutex = threading.Event(), threading.RLock()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def stop(*_):
        cancel.set(); raise KeyboardInterrupt('Stopped by user')
    for sig in handlers: signal.signal(sig, stop)
    old_entry = queue.__file__; queue.__file__ = __file__
    reports = {p.parent.name.removeprefix('report_'): ab.read(p)['result'] for p in output.glob('report_*/complete.json')}
    failures, status, error = {}, 'running', None
    score_status = 'blocked_metadata' if metadata_error else 'pending'
    score_rows_path = output / 'score_rows.json'

    def flush():
        with mutex:
            workers = {}
            for p in output.glob('*/attempt_*/queue_worker.json'):
                workers[str(p.parent.parent.name) + '/' + p.parent.name] = ab.read(p)
            result = dict(status=status, error=error, failures=failures, score_status=score_status,
                metadata_error=metadata_error, completed=len(reports), expected=24, results=reports,
                workers=workers, deadline=deadline, **summary(reports),
                note='VOC val development; incomplete groups have no mean advantage; no significance or deployment.')
            ab.write(output / 'summary.json', result)
            ab.write(output / 'status.json', {k: result[k] for k in ('status','error','completed','expected','score_status','deadline')})
            lines = ['| Training / alpha | Complete | Mean mIoU % | Sample SD pp |', '|---|---|---:|---:|']
            for key, row in result['groups'].items():
                fmt = lambda v: '—' if v is None else f'{100*v:.4f}'
                lines.append(f'| {key} | {row["complete"]} | {fmt(row["mean_miou"])} | {fmt(row["sample_std_miou"])} |')
            (output / 'summary.md').write_text(f'Status: {status}; {len(reports)}/24. Score branch: {score_status}.\n\n'
                + '\n'.join(lines) + '\n\nSee summary.json for paired contrasts and worker timing.\n', encoding='utf-8')

    def execute(key, stage, gpu, mode='original', seed=42, alpha=.2, model=None, ids=None, accepted=False):
        def action(attempt):
            job = dict(common, raw_signature=signature, group='C', variant='d2', stage=stage,
                attempt=str(attempt), training_mode=mode, seed=seed, alpha=alpha, deadline=deadline,
                score_tag=score_tag, recipe=recipe, score_root=str(score_root), plan=plan,
                score_identity_hash=object_hash(score_identity), score_ids=ids)
            env = ab.base_env(gpu)
            env.update(OVERRIDE_SAM3_CKPT=common['checkpoint'], OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT=str(alpha))
            artifacts, values = [attempt], {}
            if stage == 'train':
                w, b = weights / str(seed) / attempt.name, banks / str(seed) / attempt.name
                job.update(raw=raw, weights=str(w), bank=str(b),
                    split_file=str(Path(refs[42]['weights']) / 'training_report.json'),
                    score_rows=str(score_rows_path), score_rows_sha256=ab.digest(score_rows_path))
                values.update(weights=str(w), bank=str(b)); artifacts += [w,b]
            elif stage == 'eval':
                job.update(weights=model['weights'], bank=model['bank'], cleanup='PRE_CLEANED', metrics=str(attempt / 'metrics.json'))
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(model['weights']) / 'adapter.pt'),
                    OVERRIDE_INF_FEATURE_BANK_ROOT=model['bank'], OVERRIDE_INF_PROPOSALS_DIR=common['proposals'])
                values.update(metrics=job['metrics'], per_image=str(attempt / 'per_image_confusion.npz'))
            log = logs / f'{key}_{attempt.name}.log'
            queue.launch(job, attempt, env, log, cancel, deadline, respect_deadline=not accepted)
            report = ab.read(attempt / 'queue_worker.json')
            if report['status'] != 'complete': raise queue.GlobalFailure('Worker did not complete')
            if stage == 'train':
                tr = ab.read(Path(values['weights']) / 'training_report.json')
                if any(tr[k] != split[k] for k in split) or tr['score_training'] != score_tag:
                    raise queue.GlobalFailure('Split or confidence recipe changed')
                baseline_args = ab.read(Path(refs[seed]['weights']) / 'training_report.json')['args']
                ignored = {'output_dir', 'export_bank', 'split_file'}
                if ({k:v for k,v in tr['args'].items() if k not in ignored}
                        != {k:v for k,v in baseline_args.items() if k not in ignored}):
                    raise queue.GlobalFailure('Effective training arguments changed beyond confidence CE')
                sig = ab.read(Path(values['bank']) / 'meta/feature_signature.json')
                if sig['adapter']['type'] != PROTOCOL or sig['adapter']['architecture']['score_training'] != score_tag:
                    raise queue.GlobalFailure('Export signature mismatch')
                if ab.bank_members(values['bank']) != ab.bank_members(raw): raise queue.GlobalFailure('Export members changed')
            elif stage == 'eval':
                metric = ab.read(values['metrics'])
                validate_eval(metric, ab.read(refs[seed]['metrics']), common['val_ids'], clean['count'], alpha)
                seeds.validate_image_matrices(values['per_image'], metric)
                if metric['feature_signature'] != ab.read(Path(model['bank']) / 'meta/feature_signature.json'):
                    raise queue.GlobalFailure('Evaluated model and bank mismatch')
            values['queue_worker'] = str(attempt / 'queue_worker.json')
            return values, artifacts + [log]
        return stages.run(key, action)

    def evaluate(item, gpu, model=None, accepted=False):
        mode, seed, alpha = item
        key = cell(mode, seed, alpha)
        if key in reports: return
        model = model or refs[seed]
        result = execute('eval_' + key, 'eval', gpu, mode, seed, alpha, model, accepted=accepted)
        metric = ab.read(result['metrics']); reference = ab.read(refs[seed]['metrics'])
        if mode == 'original' and alpha == .2:
            mpsa.require_baseline_replay(reference, metric)
            require_image_replay(refs[seed]['per_image'], result['per_image'])
        value = dict(training=mode, seed=seed, alpha=alpha, **metric,
            delta_pp=100*(metric['mIoU']-reference['mIoU']), per_image=result['per_image'],
            per_class_delta_pp={c:100*(v-reference['per_class_iou'][c]) for c,v in metric['per_class_iou'].items()})
        def commit(attempt):
            path = output / f'comparison_{key}.json'; ab.write(path, value)
            return value, [path]
        saved = stages.run('report_' + key, commit)
        with mutex: reports[key] = saved
        flush()

    def fail(item, exc):
        with mutex: failures[str(item)] = dict(error=str(exc), type=type(exc).__name__)
        flush()

    def fatal(item, exc):
        raise queue.GlobalFailure(f'Historical replay failed ({item}): {exc}')

    def weighted_task(seed, gpu):
        # Once accepted, finish training and its four full evaluations past deadline.
        model = execute(f'train_weighted_{seed}', 'train', gpu, 'weighted', seed, accepted=True)
        for alpha in ALPHAS:
            try: evaluate(('weighted',seed,alpha), gpu, model, accepted=True)
            except queue.TaskFailure as exc: fail(('weighted',seed,alpha), exc)

    try:
        ab.write(output / 'artifacts.json', dict(owned_roots=list(map(str,owned)),
            read_only_sources=[str(opts.previous.resolve()),raw,common['cache'],common['pseudo'],common['proposals']]
                + ([str(score_root)] if opts.train_score_cache else []), automatic_deletion=False))
        flush()
        queue.run_round([('original',s,.2) for s in SEEDS if cell('original',s,.2) not in reports],
                        opts.gpus, evaluate, cancel, deadline, fatal)
        queue.run_round([('original',s,a) for s in SEEDS for a in ALPHAS
                         if a != .2 and cell('original',s,a) not in reports],
                        opts.gpus, evaluate, cancel, deadline, fail)
        if len(reports) == 24:
            status, score_status = 'complete', 'complete'
        elif metadata_error:
            status = 'complete_with_blocked_training'
        else:
            score_status = 'collecting'; flush()
            try:
                if not opts.train_score_cache and not (score_root / 'complete.json').exists():
                    score_root.mkdir(parents=True, exist_ok=True)
                    if (score_root / 'identity.json').exists():
                        verify_score_cache(score_root, score_identity, plan, common['train_ids'], False)
                    else:
                        if any(score_root.iterdir()): raise queue.GlobalFailure('Unidentified score cache')
                        ab.write(score_root / 'identity.json', score_identity)
                    probe = sorted(split['train_images'])[0]
                    execute('score_probe', 'scores', opts.gpus[0], ids=[probe])
                    remaining = [i for i in common['train_ids'] if i != probe]
                    queue.run_round([0,1], opts.gpus,
                        lambda shard,gpu: execute(f'score_shard_{shard}', 'scores', gpu, ids=remaining[shard::2]),
                        cancel, deadline, lambda item,exc: (_ for _ in ()).throw(ScoreReplayFailure(str(exc))))
                    if not (score_root / 'complete.json').exists():
                        # Validate all members before sealing immutable completed cache.
                        for stem in common['train_ids']:
                            if read_score_image(score_root,stem,object_hash(score_identity),plan[stem]) is None:
                                raise queue.GlobalFailure('Missing score image at commit')
                        ab.write(score_root / 'complete.json', dict(identity=object_hash(score_identity),
                            artifacts=ab.identities([score_root / f'{i}.json' for i in common['train_ids']])))
                score_rows = verify_score_cache(score_root,score_identity,plan,common['train_ids'])
            except queue.TaskFailure as exc:
                score_status = 'blocked_replay'; error = str(exc)
                status = 'complete_with_blocked_training'
            else:
                score_status = 'complete'
                def commit_scores(attempt):
                    ab.write(score_rows_path, dict(identity=object_hash(score_identity), rows=score_rows))
                    return dict(rows=str(score_rows_path)), [score_rows_path,score_root]
                stages.run('score_table', commit_scores)
                score_tag = dict(score_tag, rows_sha256=ab.digest(score_rows_path))
                flush()
                queue.run_round([s for s in SEEDS if any(cell('weighted',s,a) not in reports for a in ALPHAS)],
                                opts.gpus, weighted_task, cancel, deadline, fail)
                status = 'complete' if len(reports)==24 else 'complete_with_failures'
        stages.verify_all(); gmp.verify(records, 'Historical inputs')
        if ab.source_identity() != code: raise queue.GlobalFailure('Source changed during experiment')
    except queue.Deferred as exc:
        status, error = 'deferred', str(exc)
    except BaseException as exc:
        cancel.set(); status, error = 'incomplete', str(exc); raise
    finally:
        try: flush()
        finally:
            queue.__file__ = old_entry
            for sig, handler in handlers.items(): signal.signal(sig, handler)
            lock.unlink()
    print(f'[{status}] {output}/summary.md', flush=True)


if __name__ == '__main__':
    main()

