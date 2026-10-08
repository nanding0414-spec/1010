#!/usr/bin/env python3
"""Isolated SMFA adapter ablations using the completed combo run's SMFA cache.

No pseudo-label generation, token extraction, filtering or proposal cleaning.
Historical D3 checkpoints are replayed before any of the 21 new trainings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_mpsa_combo_queue as queue
from tools import experimental_pmfa_ab as ab
from tools import experimental_gmp_ab as gmp
from tools import experimental_gmp_seeds as seeds
from tools import experimental_mpsa_ab as mpsa

SEEDS = (42, 123, 2026)
FORMAT = 'smfa_mlp_ablation_v1'


@dataclass(frozen=True)
class Architecture:
    depth: int
    width: int = 512
    residual: bool = False
    layernorm: bool = False

    def __post_init__(self):
        if self.depth not in (2, 3, 4, 5) or self.width not in (512, 704):
            raise ValueError('Unsupported fixed architecture')
        if type(self.residual) is not bool or type(self.layernorm) is not bool:
            raise ValueError('Architecture flags must be boolean')


RECIPES = {
    'd3': Architecture(3),
    'd2': Architecture(2),
    'd4': Architecture(4),
    'd5': Architecture(5),
    'res4': Architecture(4, residual=True),
    'res5': Architecture(5, residual=True),
    'ln3': Architecture(3, layernorm=True),
    'wide3': Architecture(3, width=704),
}
NEW = tuple(k for k in RECIPES if k != 'd3')


def parameter_count(recipe, input_dim=1280, output_dim=256):
    w = recipe.width
    return ((input_dim + 1) * w + (recipe.depth - 2) * (w + 1) * w
            + (w + 1) * output_dim + (2 * w * (recipe.depth - 1) if recipe.layernorm else 0))


def adapter_tag(variant):
    if variant not in RECIPES:
        raise ValueError(f'Unknown architecture: {variant}')
    return dict(format=FORMAT, variant=variant, **asdict(RECIPES[variant]))


def make_adapter_class(legacy=None):
    """Same plain-layer construction/init order as legacy; residuals are hidden-only."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Hidden(nn.Module):
        def __init__(self, width, residual, layernorm):
            super().__init__()
            self.linear = nn.Linear(width, width)
            self.norm = nn.LayerNorm(width) if layernorm else nn.Identity()
            self.residual = residual

        def forward(self, x):
            delta = F.gelu(self.norm(self.linear(x)))
            return x + delta if self.residual else delta

    class Adapter(nn.Module):
        def __init__(self, input_dim, hidden_dim=512, output_dim=256, num_layers=3,
                     experimental_mlp=None):
            super().__init__()
            if experimental_mlp is None:
                raise ValueError('Experimental adapter requires a complete architecture tag')
            tag = dict(experimental_mlp)
            variant = tag.get('variant')
            if tag != adapter_tag(variant):
                raise ValueError('Adapter architecture tag differs from the fixed protocol')
            recipe = RECIPES[variant]
            if (hidden_dim, num_layers, output_dim, input_dim) != (recipe.width, recipe.depth, 256, 1280):
                raise ValueError('Adapter dimensions disagree with architecture tag')
            self.architecture = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, num_layers=num_layers, experimental_mlp=tag)
            layers = [nn.Linear(input_dim, hidden_dim)]
            if recipe.layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                if recipe.residual or recipe.layernorm:
                    layers.append(Hidden(hidden_dim, recipe.residual, recipe.layernorm))
                else:
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            if not torch.isfinite(x).all():
                raise ValueError('Nonfinite adapter input')
            return F.normalize(self.net(F.normalize(x.float(), dim=-1)), dim=-1)

    return Adapter


def install_adapter(variant):
    """Patch only this worker. Legacy loaders reject the extra architecture key."""
    import torch
    import src.feature_adapter as fa
    legacy, original_load, original_signature = fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature
    adapter = make_adapter_class(legacy)
    expected = adapter_tag(variant)

    def factory(input_dim, hidden_dim=512, output_dim=256, num_layers=3, experimental_mlp=None):
        tag = expected if experimental_mlp is None else experimental_mlp
        if tag != expected:
            raise ValueError('Checkpoint belongs to a different MLP ablation')
        return adapter(input_dim, hidden_dim, output_dim, num_layers, tag)

    def load(path, device='cpu', source_signature=None):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('architecture', {}).get('experimental_mlp') != expected:
            raise ValueError('Checkpoint architecture mismatch (including legacy checkpoints)')
        return original_load(path, device, source_signature)

    def signature(source, checkpoint_path, architecture):
        if architecture.get('experimental_mlp') != expected:
            raise ValueError('Export architecture mismatch')
        result = original_signature(source, checkpoint_path, architecture)
        result['adapter']['type'] = FORMAT
        return result

    fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature = factory, load, signature


def training_argv(argv, seed, variant):
    if seed not in SEEDS or variant not in NEW:
        raise ValueError('Only predeclared new SMFA adapter jobs may train')
    result = seeds.training_argv(argv, seed)
    recipe = RECIPES[variant]
    for flag, value in (('--num-layers', recipe.depth), ('--hidden-dim', recipe.width)):
        if result.count(flag) != 1:
            raise ValueError(f'Missing/duplicate {flag}')
        result[result.index(flag) + 1] = str(value)
    return result


def worker(job):
    if job['stage'] not in ('train', 'eval') or job.get('group') != 'SMFA':
        raise ValueError('This worker only trains/evaluates SMFA; no build/extract')
    variant = job['variant']
    if variant not in RECIPES or (variant == 'd3' and job['stage'] != 'eval'):
        raise ValueError('D3 is read-only; unknown architecture')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'sam3-main'))
    import torch
    started = time.monotonic()
    status, category, error = 'failed', 'local', None
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    try:
        if variant != 'd3':
            install_adapter(variant)
        if job['stage'] == 'train':
            original_run = ab.runpy.run_path
            original_add = argparse.ArgumentParser.add_argument

            def add_argument(self, *args, **kwargs):
                if '--num-layers' in args:
                    kwargs['choices'] = [2, 3, 4, 5]
                return original_add(self, *args, **kwargs)

            argparse.ArgumentParser.add_argument = add_argument

            def train(path, *args, **kwargs):
                if Path(path).resolve() != ROOT / 'tools/train_feature_adapter.py':
                    raise queue.GlobalFailure('Unexpected trainer')
                sys.argv = training_argv(sys.argv, job['seed'], variant)
                ab.write(Path(job['attempt']) / 'effective_training_argv.json',
                         dict(argv=sys.argv, seed=job['seed'], architecture=adapter_tag(variant)))
                return original_run(path, *args, **kwargs)

            ab.runpy.run_path = train

        def install_pooling(mode, *_):
            if mode != 'smfa':
                raise ValueError('Non-SMFA method refused')
            import src.feature_signature as fs
            fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
            if job['stage'] == 'eval':
                # Only the SMFA column is read. Old A-F vectors are never used.
                queue.install_cached_eval(job)

        ab.install_pooling = install_pooling
        job['mode'] = 'smfa'
        job['pmfa'] = asdict(ab.PMFA())
        ab.worker(job)
        torch.cuda.synchronize()
        status = 'complete'
    except torch.cuda.OutOfMemoryError as exc:
        category, error = 'oom', str(exc)
        raise
    except (ValueError, KeyError, FileNotFoundError, queue.GlobalFailure) as exc:
        category, error = 'global', str(exc)
        raise
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        ab.write(Path(job['attempt']) / 'queue_worker.json', dict(status=status, category=category,
            error=error, seconds=time.monotonic() - started, variant=variant, seed=job['seed'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            precision_policy=ab.POLICY, feature_extraction='reused historical SMFA vectors'))


def verified_history(folder, clean_dir):
    """Bind the selected SMFA replay jobs and cache to the completed parent run."""
    import numpy as np
    folder = Path(folder).resolve()
    manifest = ab.read(folder / 'manifest.json')
    if (manifest.get('protocol') != 'mpsa_combo_queue_v1'
            or ab.read(folder / 'status.json').get('status') != 'complete'):
        raise ValueError('A completed original combo queue is required as the SMFA cache source')
    print(f'[History] validating completed SMFA cache source: {folder}', flush=True)
    records = []
    for field in ('code', 'inputs', 'references'):
        gmp.verify(manifest[field], 'Historical ' + field)
        records.extend(manifest[field])
    old_paths = {r['path'] for r in manifest['code']}
    for record in ab.source_identity():
        path = Path(record['path'])
        if record['path'] not in old_paths and not (path.parent == ROOT / 'tools' and path.name.startswith('experimental_')):
            raise ValueError(f'Unverified new production code: {path}')
    records += ab.identities([folder / n for n in ('manifest.json', 'status.json', 'summary.json')])
    jobs, references, split, raw = {}, {}, None, None
    for seed in SEEDS:
        marker_path = folder / f'replay_SMFA_{seed}' / 'complete.json'
        marker = ab.read(marker_path)
        gmp.verify(marker['artifacts'], 'Historical SMFA replay')
        records += marker['artifacts'] + ab.identities([marker_path])
        result = marker['result']
        job = ab.read(Path(result['queue_worker']).parent / 'job.json')
        if job['stage'] != 'eval' or job['group'] != 'SMFA' or job['seed'] != seed:
            raise ValueError('Wrong historical replay job')
        if Path(job['proposals']).resolve() != clean_dir:
            raise ValueError('Clean proposal cache differs from the historically replayed cache')
        sig = job['raw_signature']
        if (sig != manifest['signatures']['SMFA'] or sig.get('format_version') != 3001
                or sig['experimental_pooling'] != ab.signature({}, 'smfa', ab.PMFA())['experimental_pooling']):
            raise ValueError('Not the expected original FP32 SMFA feature space')
        training_path = Path(job['weights']) / 'training_report.json'
        training = ab.read(training_path)
        current_raw = str(Path(training['args']['source_bank']).resolve())
        current_split = {k: training[k] for k in ('train_images', 'validation_images')}
        if training['args']['seed'] != seed or training['args']['num_layers'] != 3:
            raise ValueError('Historical reference is not the requested three-layer seed')
        expected_training = dict(hidden_dim=512, output_dim=256, recipe='v1', classifier='linear',
            selection_metric='knn', eval_topk=5, epochs=50, steps_per_epoch=100,
            classes_per_batch=8, instances_per_class=4, lr=.001, temperature=.1,
            contrastive_weight=.5, feature_dropout=0., consistency_weight=0., label_smoothing=0.)
        if any(training['args'].get(k) != v for k, v in expected_training.items()):
            raise ValueError('Historical three-layer reference uses a different v1 training recipe')
        if split is not None and (split != current_split or raw != current_raw):
            raise ValueError('Reference raw bank or internal split differs across seeds')
        split, raw = current_split, current_raw
        metric = ab.read(result['metrics'])
        seeds.validate_image_matrices(result['per_image'], metric)
        if (len(job['val_ids']) != 1449 or len(set(job['val_ids'])) != 1449
                or metric['evaluated_ids'] != job['val_ids'] or metric['eval_limit'] != 0
                or metric['cleanup'] is not None):
            raise ValueError('Historical validation coverage/cleanup differs')
        if jobs and any(job[k] != jobs[42][k] for k in ('val_ids', 'raw_signature', 'cache_identity', 'cache', 'images')):
            raise ValueError('Historical query inputs differ across seeds')
        jobs[seed] = job
        references[seed] = dict(raw=raw, weights=job['weights'], bank=job['bank'],
                                metrics=result['metrics'], per_image=result['per_image'])
    common = jobs[42]
    if ab.read(Path(raw) / 'meta/feature_signature.json') != common['raw_signature']:
        raise ValueError('Raw bank signature mismatch')
    expected_space = dict(embed_dim=1280, resolution=1008, token_norm='layernorm',
                          context_subtract=0, use_bf16=False, feature_source='trunk_fpn')
    if any(common['raw_signature'].get(k) != v for k, v in expected_space.items()):
        raise ValueError('Historical SMFA geometry/normalization/precision differs')
    split_train, split_val = map(set, (split['train_images'], split['validation_images']))
    if not split_train or not split_val or split_train & split_val:
        raise ValueError('Invalid fixed training split')
    if ab.digest(common['checkpoint']) != common['checkpoint_sha256']:
        raise ValueError('Base checkpoint changed')
    from src.adapter_training import load_training_bank, split_from_report
    x, y, image_groups, classes, _, _ = load_training_bank(raw, common['train_list'])
    if len(classes) != 21 or x.shape[1] != 1280:
        raise ValueError('Expected all 21 VOC classes and 1280-d SMFA vectors')
    if len(x) != sum(len(v) for v in ab.bank_members(raw).values()):
        raise ValueError('Invalid/duplicate retained raw bank members would be silently dropped')
    train_rows, val_rows = split_from_report(image_groups, y, Path(references[42]['weights']) / 'training_report.json')
    if (set(image_groups[train_rows]) != split_train or set(image_groups[val_rows]) != split_val
            or set(y[train_rows]) != set(range(21)) or len(val_rows) == 0):
        raise ValueError('Historical fixed split no longer has exact image membership/all training classes')
    del x, y, image_groups
    # Validate all cache payloads and mask identities, not just the first image.
    print('[History] checking all 1449 SMFA query vectors and mask identities', flush=True)
    for stem in common['val_ids']:
        proposal = clean_dir / f'{stem}.npz'
        identity = common['cache_identity'][stem]
        if (ab.digest(proposal) != identity['mask_source_sha256']
                or ab.digest(common['images'][stem]) != identity['image_sha256']):
            raise ValueError(f'Changed image/proposals: {stem}')
        with np.load(proposal, allow_pickle=False) as values:
            masks = values['masks']
            vectors = queue.cache_vectors(Path(common['cache']) / 'val' / stem,
                                          identity, 'SMFA', queue.mask_hashes(masks))
            if len(vectors) != len(values['scores']):
                raise ValueError('Proposal count mismatch')
        cached = queue.cache_read(Path(common['cache']) / 'val' / stem, identity)
        records += cached['artifacts'] + ab.identities([Path(common['cache']) / 'val' / stem / 'complete.json'])
    # Explicitly bind all used train/gallery files even if their original stage is relocated.
    used_paths = ab.files_under([raw] + [r[k] for r in references.values() for k in ('weights', 'bank')])
    recorded = {r['path'] for r in records}
    if any(str(p.resolve()) not in recorded for p in used_paths):
        raise ValueError('Unrecorded files in historical SMFA bank/weights; refuse changed inputs')
    records += ab.identities(used_paths)
    return common, references, split, records, manifest['clean']


def launch(job, attempt, env, log, cancel, deadline):
    while True:
        if cancel.is_set():
            raise queue.GlobalFailure('Cancelled')
        queue.ensure_resources(deadline)
        line = subprocess.check_output(['nvidia-smi', f'--id={env["CUDA_VISIBLE_DEVICES"]}',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, util = map(float, line.strip().split(','))
        if not all(math.isfinite(x) and x >= 0 for x in (used, util)):
            raise queue.GlobalFailure('Invalid GPU telemetry')
        if used <= 1000 and util <= 10:
            break
        cancel.wait(10)
    ab.write(attempt / 'job.json', job)
    cmd = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--worker', str(attempt / 'job.json')]
    ab.write(attempt / 'command.json', dict(command=cmd,
        environment={k: v for k, v in env.items() if k.startswith('OVERRIDE_') or k == 'CUDA_VISIBLE_DEVICES'}))
    log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
            print(f'[Worker] {job["variant"]}/{job["seed"]}/{job["stage"]} GPU={env["CUDA_VISIBLE_DEVICES"]} pid={process.pid} {log}', flush=True)
            while process.poll() is None:
                if cancel.wait(1):
                    raise queue.GlobalFailure('Cancelled')
            if process.returncode:
                path = attempt / 'queue_worker.json'
                report = ab.read(path) if path.exists() else {}
                cls = {'global': queue.GlobalFailure, 'oom': queue.TaskOOM}.get(report.get('category'), queue.TaskFailure)
                raise cls(report.get('error') or f'Worker exit {process.returncode}: {log}')
    finally:
        if process is not None and process.poll() is None:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()


def validate_metrics(current, reference, ids, clean_count):
    import numpy as np
    for value in (current, reference):
        if (value['evaluated_ids'] != ids or value['evaluated_count'] != 1449 or value['eval_limit'] != 0
                or value['cleanup'] is not None or value['proposal_count_before'] != clean_count
                or value['proposal_count_after'] != clean_count):
            raise ValueError('Incomplete validation or unexpected cleaning')
        matrix = np.asarray(value['confusion_matrix'])
        if (matrix.shape != (21, 21) or not np.isfinite(matrix).all() or (matrix < 0).any()
                or not np.equal(matrix, np.floor(matrix)).all()):
            raise ValueError('Invalid confusion matrix')
        union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
        score = np.mean(matrix.diagonal() / union)
        if not math.isfinite(score) or not math.isfinite(value['mIoU']) or abs(score - value['mIoU']) > 1e-7:
            raise ValueError('mIoU disagrees with confusion matrix')
        recorded_iou = list(value['per_class_iou'].values())
        # JSON preserves the canonical evaluation class order; check means here
        # without assuming arbitrary user-facing aliases can be sorted.
        if len(recorded_iou) != 21 or not all(math.isfinite(v) and 0 <= v <= 1 for v in recorded_iou):
            raise ValueError('Invalid per-class IoU')
        if abs(float(np.mean(recorded_iou)) - score) > 1e-7:
            raise ValueError('Per-class IoU mean disagrees')
    excluded = {'INF_FEATURE_BANK_ROOT', 'INF_SAVE_DIR'}
    if ({k: v for k, v in current['inference_settings'].items() if k not in excluded}
            != {k: v for k, v in reference['inference_settings'].items() if k not in excluded}):
        raise ValueError('Inference settings changed')
    if current['proposals_dir'] != reference['proposals_dir']:
        raise ValueError('Proposal source changed')


def comparison(reference, current, variant, seed, ids, clean_count):
    a, b = ab.read(reference['metrics']), ab.read(current['metrics'])
    validate_metrics(b, a, ids, clean_count)
    return dict(status='complete', variant=variant, seed=seed, baseline_miou=a['mIoU'],
        candidate_miou=b['mIoU'], delta_pp=100 * (b['mIoU'] - a['mIoU']),
        architecture=asdict(RECIPES[variant]), adapter_parameters=parameter_count(RECIPES[variant]),
        evaluated_ids=ids, confusion_matrices=dict(baseline=a['confusion_matrix'], candidate=b['confusion_matrix']),
        per_class_iou={c: dict(baseline=v, candidate=b['per_class_iou'][c],
            delta_pp=100 * (b['per_class_iou'][c] - v)) for c, v in a['per_class_iou'].items()},
        per_image_files=dict(baseline=reference['per_image'], candidate=current['per_image']),
        note='Fixed SMFA inputs/members/split; cached evaluation excludes feature extraction; no significance claim.')


def summarize(reports):
    result = {}
    for variant in RECIPES:
        rows = [reports[f'{variant}_{seed}'] for seed in SEEDS if f'{variant}_{seed}' in reports]
        scores = [r['candidate_miou'] for r in rows]
        result[variant] = dict(complete=len(rows) == 3, seeds=[r['seed'] for r in rows],
            mean_miou=statistics.mean(scores) if scores else None,
            sample_std_miou=statistics.stdev(scores) if len(scores) > 1 else None,
            mean_delta_pp=statistics.mean(r['delta_pp'] for r in rows) if rows else None,
            parameters=parameter_count(RECIPES[variant]), architecture=asdict(RECIPES[variant]))
    return result


def paired_contrasts(reports):
    result = {}
    for left, right in [('d3', 'd2'), ('d4', 'd3'), ('d5', 'd3'), ('res4', 'd4'),
                        ('res5', 'd5'), ('ln3', 'd3'), ('wide3', 'd3'), ('d5', 'wide3')]:
        deltas = {str(s): 100 * (reports[f'{left}_{s}']['candidate_miou'] - reports[f'{right}_{s}']['candidate_miou'])
                  for s in SEEDS if f'{left}_{s}' in reports and f'{right}_{s}' in reports}
        result[f'{left}-{right}'] = dict(complete=len(deltas) == 3, delta_pp_by_seed=deltas,
            mean_delta_pp=statistics.mean(deltas.values()) if deltas else None,
            sample_std_delta_pp=statistics.stdev(deltas.values()) if len(deltas) > 1 else None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--previous', type=Path, default=Path('experiments/mpsa_combo_20260929_105543'))
    parser.add_argument('--clean-proposals-dir', type=Path, default=Path(mpsa.DEFAULT_CLEAN))
    parser.add_argument('--version', default='smfa_mlp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--max-hours', type=float, default=48)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    opts = parser.parse_args(argv)
    os.chdir(ROOT)
    if opts.worker:
        return worker(ab.read(opts.worker))
    ab.parse_args(['--version', opts.version])
    if len(set(opts.gpus)) != 2 or min(opts.gpus) < 0 or not math.isfinite(opts.max_hours) or opts.max_hours <= 0:
        parser.error('Two distinct nonnegative GPUs and a finite positive time budget are required')
    missing = [m for m in ('numpy', 'torch', 'torchvision', 'cv2', 'faiss', 'torchmetrics', 'timm', 'einops', 'iopath')
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f'Activate moduseg_sam3; missing dependencies: {missing}')
    for gpu in opts.gpus:
        ab.check_gpu(gpu)
    common, refs, split, records, clean = verified_history(opts.previous, opts.clean_proposals_dir.resolve())
    output = ROOT / 'experiments' / opts.version
    weights = ROOT / 'weights/sam3_adapter' / opts.version
    banks = ROOT / 'storage/mlp_banks/voc' / opts.version
    logs = ROOT / 'logs' / opts.version
    code = ab.source_identity()
    identity = dict(protocol=FORMAT, previous=str(opts.previous.resolve()), references=records, code=code,
        architectures={k: adapter_tag(k) for k in RECIPES}, seeds=list(SEEDS), gpus=opts.gpus,
        clean=str(opts.clean_proposals_dir.resolve()), training='unchanged_v1_50x100_fixed_image_split')
    identity = json.loads(json.dumps(identity))
    if output.exists():
        if not opts.resume or ab.read(output / 'manifest.json') != identity:
            raise ValueError('Existing version differs or needs --resume')
    elif any(p.exists() for p in (weights, banks, logs)):
        raise FileExistsError('Version already used by another output directory')
    members = sum(len(v) for v in ab.bank_members(refs[42]['raw']).values())
    estimate = members * 256 * 4 * len(NEW) * len(SEEDS) + 2 * 2**30
    if shutil.disk_usage(ROOT).free < queue.FLOOR + (0 if opts.resume else estimate):
        raise queue.Deferred(f'Insufficient disk: estimated new {estimate / 2**30:.2f} GiB plus 5 GiB reserve')
    print('[Plan] SMFA only; d3 historical replay x3; 7 new architectures x3 seeds = 21 trainings.', flush=True)
    print('[Plan] No labels, proposals, cleaning, filtering or SAM3 feature extraction.', flush=True)
    for key, recipe in RECIPES.items():
        print(f'  {key}: {asdict(recipe)}; adapter parameters={parameter_count(recipe):,}', flush=True)
    if opts.preflight_only:
        ab.Stages(output, True).verify_all()
        print('[Preflight passed] Historical cache verified; real model replay will gate new training.', flush=True)
        return
    if not output.exists():
        ab.write(output / 'manifest.json', identity)
    stages = ab.Stages(output, True)
    stages.verify_all()
    lock = output / 'launcher.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    budget = output / 'budget.json'
    if not budget.exists():
        ab.write(budget, dict(started_at=time.time()))
    deadline = ab.read(budget)['started_at'] + opts.max_hours * 3600
    cancel = threading.Event()
    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def stop(*_):
        cancel.set()
        raise KeyboardInterrupt('Stopped by user')

    for sig in previous_signals:
        signal.signal(sig, stop)
    report_lock = threading.Lock()
    reports = {p.parent.name.removeprefix('report_'): ab.read(p)['result'] for p in output.glob('report_*/complete.json')}
    failures, replay = {}, {}
    status, error = 'running', None

    def flush():
        with report_lock:
            workers = {p.parent.name: ab.read(ab.read(p)['result']['queue_worker'])
                       for p in output.glob('*/complete.json') if 'queue_worker' in ab.read(p)['result']}
            ab.write(output / 'summary.json', dict(status=status, error=error, failures=failures,
                groups=summarize(reports), contrasts=paired_contrasts(reports), results=reports, workers=workers, deadline=deadline,
                note='All variants fixed in advance. D3 weights reused; no automatic winner deployment.'))
            ab.write(output / 'status.json', dict(status=status, error=error, completed=len(reports), expected=24, deadline=deadline))
            lines = ['| Variant | Seeds | mIoU % mean | Sample SD pp | Delta pp | Parameters |',
                     '|---|---|---:|---:|---:|---:|']
            for k, row in summarize(reports).items():
                fmt = lambda v, scale=1: '—' if v is None else f'{v * scale:.4f}'
                lines.append(f'| {k} | {row["seeds"]} | {fmt(row["mean_miou"],100)} | {fmt(row["sample_std_miou"],100)} | {fmt(row["mean_delta_pp"])} | {row["parameters"]} |')
            (output / 'summary.md').write_text(f'Status: {status}; {len(reports)}/24 completed.\n\n' + '\n'.join(lines) +
                '\n\nIncomplete seed groups are provisional. Cached evaluation time excludes SAM3 encoding.\n', encoding='utf-8')

    def execute(key, kind, variant, seed, gpu, result, accepted=False):
        def action(attempt):
            job = {k: common[k] for k in ('dataset_root', 'image_dir', 'checkpoint', 'train_list', 'val_list',
                'imagelevel_json', 'gt_dir', 'images', 'train_ids', 'val_ids', 'cache', 'cache_identity',
                'checkpoint_sha256', 'raw_signature', 'proposals')}
            job.update(stage=kind, group='SMFA', variant=variant, seed=seed, attempt=str(attempt))
            env = ab.base_env(gpu)
            env['OVERRIDE_SAM3_CKPT'] = common['checkpoint']
            artifacts, values = [attempt], {}
            if kind == 'train':
                w, b = weights / f'{variant}_{seed}' / attempt.name, banks / f'{variant}_{seed}' / attempt.name
                job.update(raw=refs[42]['raw'], weights=str(w), bank=str(b),
                           split_file=str(Path(refs[42]['weights']) / 'training_report.json'))
                values.update(weights=str(w), bank=str(b))
                artifacts += [w, b]
            else:
                job.update(weights=result['weights'], bank=result['bank'], metrics=str(attempt / 'metrics.json'), cleanup='PRE_CLEANED')
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(result['weights']) / 'adapter.pt'),
                    OVERRIDE_INF_FEATURE_BANK_ROOT=result['bank'], OVERRIDE_INF_PROPOSALS_DIR=common['proposals'])
                values.update(metrics=job['metrics'], per_image=str(attempt / 'per_image_confusion.npz'))
            log = logs / f'{key}_{attempt.name}.log'
            launch(job, attempt, env, log, cancel, float('inf') if accepted else deadline)
            report = ab.read(attempt / 'queue_worker.json')
            if report['status'] != 'complete':
                raise ValueError('Worker incomplete')
            if kind == 'train':
                training = ab.read(Path(values['weights']) / 'training_report.json')
                if any(training[k] != split[k] for k in split):
                    raise ValueError('Fixed training split changed')
                effective = ab.read(attempt / 'effective_training_argv.json')
                if effective['seed'] != seed or effective['architecture'] != adapter_tag(variant):
                    raise ValueError('Effective training recipe mismatch')
                signature = ab.read(Path(values['bank']) / 'meta/feature_signature.json')
                if (signature['adapter']['type'] != FORMAT
                        or signature['adapter']['architecture']['experimental_mlp'] != adapter_tag(variant)):
                    raise ValueError('Exported architecture signature mismatch')
                if ab.bank_members(values['bank']) != ab.bank_members(refs[42]['raw']):
                    raise ValueError('Exported bank members changed')
            else:
                metric = ab.read(values['metrics'])
                seeds.validate_image_matrices(values['per_image'], metric)
                validate_metrics(metric, ab.read(refs[seed]['metrics']), common['val_ids'], clean['count'])
                if metric['feature_signature'] != ab.read(Path(result['bank']) / 'meta/feature_signature.json'):
                    raise ValueError('Evaluated adapter/bank feature space mismatch')
            values['queue_worker'] = str(attempt / 'queue_worker.json')
            return values, artifacts + [log]

        try:
            return stages.run(key, action)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise queue.GlobalFailure(str(exc)) from exc

    def commit(variant, seed, result):
        key = f'{variant}_{seed}'
        value = comparison(replay[seed], result, variant, seed, common['val_ids'], clean['count'])

        def action(attempt):
            path = output / f'comparison_{key}.json'
            ab.write(path, value)
            return value, [path]

        saved = stages.run('report_' + key, action)
        with report_lock:
            reports[key] = saved
        flush()

    def replay_task(seed, gpu):
        result = dict(refs[seed])
        result.update(execute(f'replay_d3_{seed}', 'eval', 'd3', seed, gpu, result))
        mpsa.require_baseline_replay(ab.read(refs[seed]['metrics']), ab.read(result['metrics']))
        replay[seed] = result
        commit('d3', seed, result)

    def fail(item, exc):
        with report_lock:
            failures[f'{item[0]}_{item[1]}'] = dict(error=str(exc), type=type(exc).__name__)
        flush()

    def task(item, gpu):
        variant, seed = item
        key = f'{variant}_{seed}'
        if key in reports:
            return
        failure_path = output / 'task_failures' / f'{key}.json'
        prior = ab.read(failure_path) if failure_path.exists() else dict(attempts=0)
        if prior['attempts'] >= 2 or prior.get('oom'):
            raise queue.TaskFailure('Retry budget exhausted: ' + key)
        for retry in range(prior['attempts'], 2):
            try:
                result = dict(raw=refs[42]['raw'])
                result.update(execute('train_' + key, 'train', variant, seed, gpu, result, accepted=True))
                result.update(execute('eval_' + key, 'eval', variant, seed, gpu, result, accepted=True))
                commit(variant, seed, result)
                return
            except queue.TaskFailure as exc:
                ab.write(failure_path, dict(attempts=retry + 1, oom=isinstance(exc, queue.TaskOOM), error=str(exc)))
                if isinstance(exc, queue.TaskOOM) or retry == 1:
                    raise
                queue.ensure_resources(deadline)

    try:
        ab.write(output / 'artifacts.json', dict(owned_roots=list(map(str, (output, weights, banks, logs))),
            read_only_previous=str(opts.previous.resolve()), read_only_raw=refs[42]['raw'],
            read_only_cache=common['cache'], read_only_proposals=common['proposals'], automatic_deletion=False))
        flush()
        queue.run_round(list(SEEDS), opts.gpus, replay_task, cancel, deadline,
            lambda item, exc: (_ for _ in ()).throw(queue.GlobalFailure(f'SMFA replay failed: {exc}')))
        print('[Replay passed] Three historical SMFA confusion matrices reproduced; starting new MLPs.', flush=True)
        for seed in SEEDS:
            queue.run_round([(variant, seed) for variant in NEW], opts.gpus, task, cancel, deadline, fail)
        stages.verify_all()
        gmp.verify(records, 'Read-only historical inputs/artifacts')
        if ab.source_identity() != code:
            raise queue.GlobalFailure('Source changed during run')
        status = 'complete' if len(reports) == 24 else 'complete_with_failures'
    except queue.Deferred as exc:
        status, error = 'deferred', str(exc)
    except BaseException as exc:
        cancel.set()
        status, error = 'incomplete', str(exc)
        raise
    finally:
        flush()
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)
        lock.unlink()
    print(f'[{status}] {output}/summary.md', flush=True)


if __name__ == '__main__':
    main()
