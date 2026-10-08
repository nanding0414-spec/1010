#!/usr/bin/env python3
"""Isolated MPSA 1/2/4-phase experiment; fixed members, precleaned proposals."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_pmfa_ab as ab
from tools import experimental_rpg_ab as rpg
from tools import experimental_gmp_ab as gmp
from tools import experimental_gmp_seeds as seeds

FORMAT = 3005
OFFSETS = ((0, 0), (7, 0), (0, 7), (7, 7))
PHASES = {1: (0,), 2: (0, 3), 4: (0, 1, 2, 3)}
CLEAN = dict(min_score=0.7, min_mask_area=100, min_island_area=64,
             max_hole_area=32, max_hole_fraction=0.01, nms_iou=0.85)
DEFAULT_CLEAN = '/home/zystu2025/nd/ModuSeg_SAM3/storage/proposals/voc/active_clean_20260928_151808'

# Reviewed standalone visualization utility; not imported by model/train/eval.
# Normalize line endings/BOM only for this explicit addition. Historical source
# hashes and the new run's full byte-level source manifest remain unchanged.
REVIEWED_EXTRA_TOOLS = {
    'tools/visualize_proposal_gt_diff.py': '4942b0a4a66b04a0e8d0668f566cf02be5a20f9316c33e11f0c0af75883561a7',
}


def baseline_source_view(records):
    result = []
    reviewed = {(ROOT/name).resolve(): digest for name, digest in REVIEWED_EXTRA_TOOLS.items()}
    for row in records:
        path = Path(row['path']).resolve()
        if path in reviewed:
            actual = hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
            if actual != reviewed[path]:
                raise ValueError(f'Reviewed visualization tool has different content: {path}; inspect before allowing it')
            print(f'[Baseline] verified standalone visualization addition: {path.name}', flush=True)
        else:
            result.append(row)
    return result


def checked_baseline(folder):
    # Only the old baseline's "new production file" inventory gets this view.
    # rpg.baseline still verifies every historical source/input/artifact SHA256.
    original = ab.source_identity
    try:
        ab.source_identity = lambda: baseline_source_view(original())
        return rpg.baseline(folder)
    finally:
        ab.source_identity = original


@dataclass(frozen=True)
class MPSA:
    phases: int = 4
    checkpoint_sha256: str = ''

    def __post_init__(self):
        if type(self.phases) is not int or self.phases not in PHASES:
            raise ValueError('MPSA phases must be 1, 2 or 4')
        if len(self.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.checkpoint_sha256):
            raise ValueError('An explicit checkpoint SHA256 is required')


def signature(base, mode, params):
    if mode != 'mpsa':
        raise ValueError(mode)
    return dict(base, format_version=FORMAT, experimental_pooling=dict(
        mode='mpsa', version=1, parameters=asdict(params),
        offsets=[list(OFFSETS[i]) for i in PHASES[params.phases]],
        shift_space='after_original_preprocess_1008_square',
        image_padding='reflect_left_top_crop_right_bottom_no_wrap',
        mask_sampling='floor((center-offset)*original_size/1008); outside_zero',
        centers='14*i+7', mask_weights='binary_only_no_coverage',
        fallback='nearest_valid_center_to_shifted_original_centroid; row_major_tie',
        empty_mask='zero', zero_feature='zero; nonempty_bank_zero_is_error',
        fusion='per_phase_l2_then_equal_mean_then_l2_eps1e-12',
        placement='before_region_mlp', phase_batch_size=1, mask_chunk_size=16,
        precision_policy=ab.POLICY))


def selection(mask, offset, resolution=1008, stride=14):
    """Binary center sampling, analytically shifting a nearest-resized mask.

    Never computes occupancy/coverage. A clipped tiny mask uses its original
    centroid transformed into the shifted canvas, and the nearest valid center.
    """
    import numpy as np
    m = np.asarray(mask)
    if m.ndim != 2 or min(m.shape) < 1 or not np.isfinite(m).all() or not np.isin(m, [0, 1]).all():
        raise ValueError('Expected a finite binary original-resolution mask')
    if resolution != 1008 or stride != 14 or tuple(offset) not in OFFSETS:
        raise ValueError('Only the declared 1008/14/7 geometry is supported')
    dx, dy = offset
    yy, xx = np.meshgrid(np.arange(7, resolution, stride), np.arange(7, resolution, stride), indexing='ij')
    sx, sy = xx - dx, yy - dy
    valid = (sx >= 0) & (sy >= 0) & (sx < resolution) & (sy < resolution)
    ix = np.floor(sx * m.shape[1] / resolution).astype(int).clip(0, m.shape[1]-1)
    iy = np.floor(sy * m.shape[0] / resolution).astype(int).clip(0, m.shape[0]-1)
    selected = np.flatnonzero((valid & m[iy, ix].astype(bool)).ravel())
    fallback = False
    if len(selected) == 0 and m.any():
        ys, xs = np.nonzero(m)
        cx = (xs.mean()+.5)*resolution/m.shape[1]-.5+dx
        cy = (ys.mean()+.5)*resolution/m.shape[0]-.5+dy
        distance = np.where(valid, (xx-cx)**2+(yy-cy)**2, np.inf)
        selected = np.asarray([int(distance.argmin())])
        fallback = True
    return selected, dict(support=len(selected), nearest_fallback=fallback, empty=not bool(m.any()))


def shift_input(inp, offset):
    import torch
    import torch.nn.functional as F
    if inp.shape != (1, 3, 1008, 1008) or not inp.is_floating_point() or not torch.isfinite(inp).all():
        raise ValueError('Expected finite [1,3,1008,1008] preprocessed input')
    if tuple(offset) not in OFFSETS:
        raise ValueError('Unexpected offset')
    dx, dy = offset
    return F.pad(inp, (dx, 0, dy, 0), mode='reflect')[:, :, :1008, :1008] if dx or dy else inp


def hard_mean(tokens, indices):
    import torch
    import torch.nn.functional as F
    if tokens.ndim != 2 or not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError('Expected finite [N,C] tokens')
    if len(indices) == 0:
        return tokens.new_zeros(tokens.shape[1], dtype=torch.float32)
    index = torch.as_tensor(indices, dtype=torch.long, device=tokens.device)
    if index.min() < 0 or index.max() >= len(tokens):
        raise ValueError('Invalid center indices')
    return F.normalize(tokens[index].float().mean(0), dim=0)


def extract_many(model, rgb, masks, counts=(1, 2, 4), stats=None, diagnostic=None):
    """One SAM3 forward per phase for all masks; no mask x token x C copy."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    if not counts or any(type(c) is not int or c not in PHASES for c in counts):
        raise ValueError('Invalid phase variants')
    if model.resolution != 1008 or model.context_subtract != 0 or model.use_bf16:
        raise ValueError('MPSA requires 1008, context=0, FP32')
    masks = [np.asarray(m) for m in masks]
    if any(m.shape != rgb.shape[:2] for m in masks):
        raise ValueError('Mask/image shape mismatch')
    if not masks:
        return {c: [] for c in counts}
    # Validate before expensive image forwards.
    for m in masks:
        selection(m, (0, 0))
    inp, _ = model._preprocess_image(rgb)
    active = sorted({p for c in counts for p in PHASES[c]})
    values = {}; infos = [[] for _ in masks]
    started = time.monotonic()
    with torch.inference_mode():
        for phase in active:
            tokens, grid = model._forward_tokens(shift_input(inp, OFFSETS[phase]))
            if tuple(grid) != (72, 72) or tokens.shape != (1, 5184, model.embed_dim):
                raise ValueError('Unexpected feature grid/dimension')
            if stats is not None:
                stats['image_phase_forwards'] += 1
            vectors = []
            for start in range(0, len(masks), 16):
                for index in range(start, min(start+16, len(masks))):
                    indices, info = selection(masks[index], OFFSETS[phase])
                    vectors.append(hard_mean(tokens[0], indices))
                    infos[index].append(dict(phase=phase, offset=OFFSETS[phase], **info))
                    if stats is not None:
                        stats['nearest_fallbacks'] += int(info['nearest_fallback'])
                        stats['empty_phase_masks'] += int(info['empty'])
                        stats['support_sum'] += info['support']
                        stats['region_phases'] += 1
            values[phase] = torch.stack(vectors)
            del tokens
        output = {c: list(F.normalize(torch.stack([values[p] for p in PHASES[c]]).mean(0), dim=1))
                  for c in counts}
    if stats is not None:
        stats['images'] += 1; stats['regions'] += len(masks)
        stats['feature_seconds'] += time.monotonic()-started
    if diagnostic:
        for index, phases in enumerate(infos):
            diagnostic(dict(region_index=index, phases=phases,
                phase_cosine_to_zero={str(p): float(values[0][index]@values[p][index]) for p in active}))
    return output


def install_mpsa(params, stats, job):
    import numpy as np
    from PIL import Image
    from src.feature_extractor_sam3 import SAM3FeatureExtractor as Sam
    import src.feature_signature as fs
    fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
    old_sig = Sam.feature_signature
    Sam.feature_signature = lambda self: signature(old_sig(self), 'mpsa', params)
    def batch(self, rgb, masks):
        def record(row):
            import json
            row = dict(image_id=job.get('_image_id'), **row)
            with (Path(job['attempt'])/'region_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        return [v.cpu().numpy() for v in extract_many(self, rgb, masks, (params.phases,), stats, record)[params.phases]]
    def single(self, rgb, mask):
        return extract_many(self, rgb, [mask], (params.phases,), stats)[params.phases][0]
    def regions(self, path, rows, min_mask_area=0):
        kept = [r for r in rows if r.get('mask') is not None and r['mask'].sum() >= min_mask_area]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        result = {}
        for row, vec in zip(kept, batch(self, rgb, [r['mask'] for r in kept])):
            result.setdefault(int(row['cls_id']), []).append(vec)
        return result
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('MPSA does not use coverage pooling; image and mask are required')
    Sam.extract_features_batch = batch
    Sam.extract_features = single
    Sam._extract_mask_pooling_feature = single
    Sam.extract_features_for_regions = regions
    Sam._pool_tokens = forbidden
    print(f'[MPSA] phases={params.phases}; binary centers; post-resize offsets; no SMFA fallback', flush=True)


def clean_eval_argv(argv):
    args = list(argv)
    if args.count('--cleanup-config') != 1:
        raise ValueError('Unexpected underlying evaluation CLI')
    i = args.index('--cleanup-config')
    del args[i:i+2]
    return args


def install_clean_eval(job):
    import numpy as np
    import configs.config as cfg
    cfg.DATASET_ROOT = cfg.INF_DATASET_ROOT = job['dataset_root']
    import project_utils.metrics as metrics
    evaluate = metrics.evaluate_dataset_miou
    def per_image(preds, gts, count, ignore, *args, **kwargs):
        if len(preds) != len(job['val_ids']):
            raise ValueError('Incomplete per-image evaluation')
        np.savez_compressed(Path(job['attempt'])/'per_image_confusion.npz',
            ids=np.asarray(job['val_ids']), confusions=seeds.image_confusions(preds, gts, count, ignore))
        return evaluate(preds, gts, count, ignore, *args, **kwargs)
    metrics.evaluate_dataset_miou = per_image
    import run_inference as inference
    inference.load_eval_samples = lambda _: [dict(id=i, img=job['images'][i],
        gt=str(Path(job['gt_dir'])/f'{i}.png')) for i in job['val_ids']]
    load = inference.load_strict_proposals
    def load_image(directory, image_id, height, width):
        job['_image_id'] = image_id
        return load(directory, image_id, height, width)
    inference.load_strict_proposals = load_image
    original = inference.main
    inference.main = lambda argv: original(clean_eval_argv(argv))


def extract_worker(job):
    import numpy as np
    import torch
    from PIL import Image
    os.chdir(ROOT); sys.path.insert(0, str(ROOT/'sam3-main'))
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    ab.install_precision()  # Before the constructor's real dummy feature probe.
    import configs.config as cfg
    cfg.DATASET_ROOT = cfg.INF_DATASET_ROOT = job['dataset_root']
    from project_utils.seed_utils import set_global_seed
    set_global_seed(42)
    from src.feature_extractor_factory import build_feature_extractor
    from src.region_extractors import get_regions_from_pseudo
    stats = Counter(); started = time.monotonic(); status = 'failed'
    try:
        model = build_feature_extractor(torch.device('cuda:0'), use_interactivity=False)
        source = model.feature_signature()
        baseline_sig = ab.read(Path(job['baseline']['raw'])/'meta/feature_signature.json')
        if any(source[k] != baseline_sig.get(k) for k in source if k != 'format_version'):
            raise ValueError('Underlying SAM3 feature signature changed')
        plan = ab.read(job['plan'])
        for stem in job['ids']:
            folder = Path(job['cache'])/stem
            cached = gmp.image_cache(folder)
            identity = dict(source=source, checkpoint_sha256=job['checkpoint_sha256'],
                algorithm=signature({}, 'mpsa', MPSA(4, job['checkpoint_sha256'])))
            if cached:
                if cached['identity'] != identity:
                    raise ValueError('Image cache identity differs')
                print(f'[Image resume] {stem}', flush=True); continue
            rows = plan[stem]
            regions = get_regions_from_pseudo(job['images'][stem], 300, job['pseudo'])
            selected = []
            for row in rows:
                region = regions[row['index']]
                if hashlib.sha256(region['mask'].tobytes()).hexdigest() != row['mask_sha256']:
                    raise ValueError('Fixed member mask changed')
                selected.append(region['mask'])
            folder.mkdir(parents=True, exist_ok=True)
            number = 1
            while (folder/f'attempt_{number:03d}').exists(): number += 1
            attempt = folder/f'attempt_{number:03d}'; attempt.mkdir()
            with Image.open(job['images'][stem]) as im:
                rgb = np.asarray(im.convert('RGB'))
            diagnostics = []; image_start = time.monotonic()
            vectors = extract_many(model, rgb, selected, stats=stats, diagnostic=diagnostics.append)
            arrays = {}
            for count, values in vectors.items():
                arr = np.asarray([v.cpu().numpy() for v in values], dtype=np.float32).reshape(-1, model.embed_dim)
                if not np.isfinite(arr).all() or (np.linalg.norm(arr, axis=1) < 1e-6).any():
                    raise ValueError('Nonempty retained region has zero/invalid MPSA feature')
                arrays[f'features_{count}'] = arr
            file = attempt/'features.npz'
            np.savez_compressed(file, ids=np.asarray([r['id'] for r in rows], dtype=str),
                classes=np.asarray([r['cls'] for r in rows], dtype=str), **arrays)
            ab.write(attempt/'diagnostics.json', dict(image=stem, seconds=time.monotonic()-image_start,
                regions=diagnostics))
            ab.write(folder/'complete.json', dict(result=dict(file=str(file), identity=identity),
                artifacts=ab.identities([file, attempt/'diagnostics.json'])))
            print(f'[Extract] {stem}: {len(rows)} fixed regions; phases 1/2/4 cached', flush=True)
        torch.cuda.synchronize(); status = 'complete'
    finally:
        ab.write(Path(job['attempt'])/'worker_report.json', dict(status=status, seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            pooling_stats=dict(stats), precision_policy=ab.POLICY))


def worker(job):
    if job['stage'] in ('probe', 'extract'):
        return extract_worker(job)
    if job['stage'] not in ('train', 'eval'):
        raise ValueError('Unsupported worker stage')
    original_install = ab.install_pooling
    if job['mode'] == 'mpsa':
        ab.PMFA = MPSA; ab.EXPERIMENT_FORMAT = FORMAT; ab.signature = signature
    elif job['mode'] != 'smfa' or job['stage'] != 'eval':
        raise ValueError('SMFA is evaluation only')
    def install(mode, params, stats):
        if mode == 'mpsa': install_mpsa(params, stats, job)
        else: original_install(mode, params, stats)
        if job['stage'] == 'eval': install_clean_eval(job)
    ab.install_pooling = install
    ab.worker(job)


def validate_clean_cache(directory, ids, original, expected_count):
    """Verify exporter provenance; never rerun mask cleanup."""
    import numpy as np
    meta = ab.read(Path(directory)/'cleanup_config.json')
    if meta['config'] != CLEAN or meta['errors'] != 0 or meta['total'] != len(ids):
        raise ValueError('Clean cache recipe/count/errors mismatch')
    if Path(meta['input_dir']).resolve() != Path(original).resolve():
        raise ValueError('Clean cache was not exported from the baseline proposals')
    rows = meta['files']
    if len(rows) != len(ids) or {r['name'] for r in rows} != {i+'.npz' for i in ids}:
        raise ValueError('Clean cache provenance IDs differ')
    if {p.stem for p in Path(directory).glob('*.npz')} != set(ids):
        raise ValueError('Clean cache file IDs differ')
    total = 0
    for row in rows:
        src = Path(original)/row['name']
        if ab.digest(src) != row['source_sha256']:
            raise ValueError('Original proposal hash mismatch')
        with np.load(src, allow_pickle=False) as raw, np.load(Path(directory)/row['name'], allow_pickle=False) as clean:
            masks, scores, indices = clean['masks'], clean['scores'], clean['source_indices']
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(set(indices.tolist())) != len(indices):
                raise ValueError('Invalid source indices')
            if len(indices) != len(scores) or (indices < 0).any() or (indices >= len(raw['scores'])).any():
                raise ValueError('Out-of-range source indices')
            if masks.shape != (len(scores), *raw['masks'].shape[1:]) or not np.isin(masks, [0, 1]).all():
                raise ValueError('Invalid cleaned masks')
            if not np.array_equal(scores, raw['scores'][indices]) or not np.isfinite(scores).all():
                raise ValueError('Cleaned scores differ from original scores')
            total += len(scores)
    if total != expected_count:
        raise ValueError('Clean proposal count differs from historical evaluation')
    return dict(directory=str(Path(directory).resolve()), count=total, config=CLEAN,
                mode='precleaned_no_runtime_cleanup', metadata=str(Path(directory)/'cleanup_config.json'))


def merge(cache, ids, raw, baseline_raw, params):
    import numpy as np
    from src.feature_bank_storage import save_feature_matrix_npz
    wanted = gmp.read_bank(baseline_raw); features = {}; identity = None
    for stem in ids:
        value = gmp.image_cache(Path(cache)/stem)
        if value is None: raise ValueError(f'Missing image cache: {stem}')
        if identity is not None and identity != value['identity']: raise ValueError('Mixed image identities')
        identity = value['identity']
        with np.load(value['file'], allow_pickle=False) as data:
            for cls, sid, vec in zip(data['classes'], data['ids'], data[f'features_{params.phases}']):
                key = (str(cls), str(sid))
                if key in features: raise ValueError('Duplicate member')
                features[key] = vec
    if set(features) != set(wanted): raise ValueError('Fixed bank members differ')
    if identity['checkpoint_sha256'] != params.checkpoint_sha256: raise ValueError('Checkpoint differs')
    raw.mkdir(parents=True, exist_ok=False)
    for cls in sorted({k[0] for k in wanted}):
        keys = sorted(k for k in wanted if k[0] == cls)
        save_feature_matrix_npz(raw/'features'/cls/'features.npz', np.stack([features[k] for k in keys]), [k[1] for k in keys])
    ab.write(raw/'meta/feature_signature.json', signature(identity['source'], 'mpsa', params))
    shutil.copy2(Path(baseline_raw)/'meta/class_mapping.json', raw/'meta/class_mapping.json')
    ab.write(raw/'meta/fixed_members.json', dict(source=str(baseline_raw), count=len(features), filtering='no_refilter'))


def require_baseline_replay(old, current):
    import numpy as np
    if current['cleanup'] is not None or current['proposal_count_before'] != current['proposal_count_after']:
        raise ValueError('Precleaned evaluation unexpectedly cleaned proposals')
    if (old['evaluated_ids'] != current['evaluated_ids'] or
        old['proposal_count_after'] != current['proposal_count_after'] or
        not np.array_equal(old['confusion_matrix'], current['confusion_matrix'])):
        raise ValueError('SMFA clean-cache replay differs from historical result; stop before MPSA training')


def compare(base, current, ids, count):
    value = gmp.report(base, current, ids)
    value['mpsa_miou'] = value.pop('gmp_miou')
    value['confusion_matrices']['mpsa'] = value['confusion_matrices'].pop('gmp')
    for row in value['per_class_iou'].values(): row['mpsa'] = row.pop('gmp')
    value.update(phases=count, seed=42, note='Fixed members; all three predeclared variants fully evaluated; no val selection.')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--previous', type=Path, default=Path('experiments/pmfa_fixed_20260924_101553'))
    parser.add_argument('--clean-proposals-dir', type=Path, default=Path(DEFAULT_CLEAN))
    parser.add_argument('--version', default='mpsa_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    opts = parser.parse_args(argv); os.chdir(ROOT)
    if opts.worker: return worker(ab.read(opts.worker))
    if len(set(opts.gpus)) != 2 or min(opts.gpus) < 0: parser.error('Two distinct physical GPUs required')
    ab.parse_args(['--version', opts.version])
    opts.previous = opts.previous.resolve(); opts.clean_proposals_dir = opts.clean_proposals_dir.resolve()
    rpg.verify_records = gmp.verify
    old, base, records = checked_baseline(opts.previous)
    flags = ['--version', opts.version, '--gpu', str(opts.gpus[0])]
    for key in ('dataset_root', 'image_dir', 'checkpoint', 'train_list', 'val_list', 'imagelevel_json', 'gt_dir'):
        flags += ['--'+key.replace('_', '-'), old['options'][key]]
    pseudo = old['selected'].get('pseudo') or ab.read(opts.previous/'pseudo/complete.json')['result']['pseudo']
    original = old['selected'].get('proposals') or ab.read(opts.previous/'proposals/complete.json')['result']['proposals']
    args = ab.parse_args(flags+['--pseudo-dir', pseudo, '--proposals-dir', str(opts.clean_proposals_dir)])
    context = ab.preflight(args); ab.check_gpu(opts.gpus[1])
    if not all(context['selected'].values()): raise ValueError('Inputs incomplete; no regeneration allowed')
    historic = ab.read(base['metrics'])
    clean = validate_clean_cache(opts.clean_proposals_dir, context['val_ids'], original, historic['proposal_count_after'])
    context['inputs'] += ab.identities([opts.clean_proposals_dir/'cleanup_config.json'])
    checkpoint_hash = ab.digest(args.checkpoint)
    output = ROOT/'experiments'/opts.version; cache = ROOT/'storage/raw_banks/voc'/opts.version/'image_cache'
    logs = ROOT/'logs'/opts.version; weights_root = ROOT/'weights/sam3_adapter'/opts.version
    bank_root = ROOT/'storage/mlp_banks/voc'/opts.version
    identity = dict(inputs=context['inputs'], code=context['code'], baseline=records,
        previous=str(opts.previous), clean=clean, gpus=opts.gpus, params=[asdict(MPSA(c, checkpoint_hash)) for c in PHASES],
        protocol='fixed_members_mpsa_1_2_4_fp32_seed42_v1')
    if output.exists():
        if not opts.resume or ab.read(output/'manifest.json') != identity: raise ValueError('Version differs or --resume missing')
    elif any(p.exists() for p in (cache.parent, logs, weights_root, bank_root)):
        raise FileExistsError('Version already used')
    if opts.preflight_only:
        print('[Preflight passed] Precleaned proposals; SMFA eval only; shared 4-phase extraction; new MLPs for 1/2/4.', flush=True)
        return
    if not output.exists(): ab.write(output/'manifest.json', identity)
    lock = output/'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    cancel = threading.Event(); previous_signal = signal.getsignal(signal.SIGTERM)
    def stop(*_): cancel.set(); raise KeyboardInterrupt('MPSA interrupted')
    signal.signal(signal.SIGTERM, stop); ab.SCRIPT = Path(__file__).resolve()
    stages = ab.Stages(output, opts.resume)
    common = {k: str(getattr(args, k)) for k in ('dataset_root', 'image_dir', 'imagelevel_json', 'checkpoint', 'train_list', 'val_list', 'gt_dir')}
    common.update(images=context['images'], train_ids=context['train_ids'], val_ids=context['val_ids'],
        baseline=base, pseudo=pseudo, cache=str(cache), plan=str(output/'members.json'), checkpoint_sha256=checkpoint_hash)

    def execute(key, kind, gpu, mode='mpsa', count=4, **extras):
        if cancel.is_set(): raise RuntimeError('Cancelled')
        def action(attempt):
            job = dict(common, attempt=str(attempt), stage=kind, mode=mode,
                pmfa=asdict(MPSA(count, checkpoint_hash)) if mode == 'mpsa' else asdict(ab.PMFA()), **extras)
            env = ab.base_env(gpu); env['OVERRIDE_SAM3_CKPT'] = str(args.checkpoint)
            result = {}; artifacts = [attempt]
            if kind == 'train':
                w = weights_root/key/attempt.name; b = bank_root/key/attempt.name
                job.update(weights=str(w), bank=str(b), split_file=str(Path(base['weights'])/'training_report.json'))
                result.update(weights=str(w), bank=str(b)); artifacts += [w, b]
            if kind == 'eval':
                job.update(metrics=str(attempt/'metrics.json'), cleanup='PRE_CLEANED_REMOVE_THIS_ARGUMENT')
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(extras['weights'])/'adapter.pt'),
                    OVERRIDE_INF_FEATURE_BANK_ROOT=extras['bank'], OVERRIDE_INF_PROPOSALS_DIR=str(opts.clean_proposals_dir))
                result.update(metrics=job['metrics'], per_image=str(attempt/'per_image_confusion.npz'))
            log = logs/f'{key}_{attempt.name}.log'
            ab.launch(job, attempt, env, log, cancel=cancel)
            if ab.read(attempt/'worker_report.json')['status'] != 'complete': raise RuntimeError('Incomplete worker')
            if kind == 'eval':
                metrics = ab.read(job['metrics'])
                if metrics['cleanup'] is not None or metrics['proposal_count_before'] != clean['count'] or metrics['proposal_count_after'] != clean['count']:
                    raise ValueError('Unexpected cleanup/proposal count')
                seeds.validate_image_matrices(result['per_image'], metrics)
            if kind == 'train':
                training = ab.read(Path(result['weights'])/'training_report.json')
                split = ab.read(Path(base['weights'])/'training_report.json')
                if any(training[k] != split[k] for k in ('train_images', 'validation_images')):
                    raise ValueError('Internal split changed')
            result['worker_report'] = str(attempt/'worker_report.json')
            return result, artifacts+[log]
        return stages.run(key, action)

    try:
        ab.write(output/'artifacts.json', dict(owned_roots=list(map(str, [output, cache.parent, logs, weights_root, bank_root])),
            read_only_inputs=context['selected'], baseline=str(opts.previous), automatic_deletion=False))
        ab.write(output/'proposal_policy.json', clean)
        replay = dict(base)
        replay.update(execute('smfa_clean_eval', 'eval', opts.gpus[0], mode='smfa', bank=base['bank'], weights=base['weights']))
        require_baseline_replay(historic, ab.read(replay['metrics']))
        def membership(attempt):
            for key, val in ab.base_env(opts.gpus[0]).items():
                if key.startswith('OVERRIDE_'): os.environ[key] = val
            plan = gmp.enumerate_members(context['train_ids'], context['images'], pseudo, base['raw'])
            ab.write(output/'members.json', plan)
            return {}, [output/'members.json']
        stages.run('membership', membership)
        split = ab.read(Path(base['weights'])/'training_report.json')
        probe_ids = random.Random(42).sample(sorted(split['train_images']), min(8, len(split['train_images'])))
        execute('probe', 'probe', opts.gpus[0], ids=probe_ids)
        probe_rows = [ab.read(Path(gmp.image_cache(cache/stem)['file']).parent/'diagnostics.json') for stem in probe_ids]
        if not any(row['regions'] for row in probe_rows): raise ValueError('Probe contains no retained regions')
        ab.write(output/'probe_summary.json', dict(ids=probe_ids,
            mean_image_seconds=sum(row['seconds'] for row in probe_rows)/len(probe_rows),
            max_image_seconds=max(row['seconds'] for row in probe_rows),
            note='Four sequential FP32 forwards per image; probe features reused during full extraction.'))
        gmp.parallel([lambda i=i: execute(f'extract_{i}', 'extract', opts.gpus[i], ids=context['train_ids'][i::2])
                      for i in range(2)], cancel)
        candidates = {}
        for count in PHASES:
            def merge_action(attempt, count=count):
                raw = cache.parent/f'phase{count}'/attempt.name
                merge(cache, context['train_ids'], raw, base['raw'], MPSA(count, checkpoint_hash))
                return dict(raw=str(raw)), [raw]
            candidates[count] = stages.run(f'merge_{count}', merge_action)
        reports = {}
        def lane(counts, gpu):
            for count in counts:
                current = candidates[count]
                current.update(execute(f'train_{count}', 'train', gpu, count=count, raw=current['raw']))
                current.update(execute(f'eval_{count}', 'eval', gpu, count=count, bank=current['bank'], weights=current['weights']))
                value = compare(replay, current, context['val_ids'], count)
                value['per_image_files'] = dict(smfa=replay['per_image'], mpsa=current['per_image'])
                reports[count] = value
                ab.write(output/f'comparison_phase{count}.json', value)
        gmp.parallel([lambda: lane((1, 2), opts.gpus[0]), lambda: lane((4,), opts.gpus[1])], cancel)
        stages.verify_all(); gmp.verify(records, 'Baseline'); gmp.verify(context['inputs'], 'Inputs')
        if ab.source_identity() != context['code']: raise ValueError('Source changed during experiment')
        workers = {p.parent.name: ab.read(ab.read(p)['result']['worker_report'])
            for p in output.glob('*/complete.json') if 'worker_report' in ab.read(p)['result']}
        ab.write(output/'summary.json', dict(status='complete', seed=42, proposal_policy=clean,
            results={str(k): v for k, v in sorted(reports.items())}, workers=workers,
            phase4_minus_phase1_pp=100*(reports[4]['mpsa_miou']-reports[1]['mpsa_miou']),
            note='Single seed; no statistical significance claim; no validation-driven parameter search.'))
        (output/'summary.csv').write_text('phases,smfa_miou,mpsa_miou,delta_pp\n'+''.join(
            f'{c},{reports[c]["smfa_miou"]},{reports[c]["mpsa_miou"]},{reports[c]["delta_pp"]}\n' for c in sorted(reports)), encoding='utf-8')
        ab.write(output/'status.json', dict(status='complete'))
        print(f'[DONE] {output}/summary.json', flush=True)
    except BaseException as exc:
        cancel.set(); ab.write(output/'status.json', dict(status='incomplete', error=str(exc))); raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal); lock.unlink()


if __name__ == '__main__':
    main()
