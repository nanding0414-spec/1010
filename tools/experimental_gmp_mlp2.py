#!/usr/bin/env python3
"""Complete the fixed SMFA/GMP x two/three-layer comparison with 3 new trainings."""
from __future__ import annotations
import argparse
from datetime import datetime
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
from tools import experimental_smfa_mlp_queue as mlp
from tools import experimental_mpsa_combo_queue as queue
ab, gmp, mpsa, seeds = mlp.ab, mlp.gmp, mlp.mpsa, mlp.seeds
SEEDS = mlp.SEEDS
PROTOCOL = 'hard_center_gmp_rho01_mlp2_v1'


def depth_two_argv(argv):
    result = list(argv)
    if result.count('--num-layers') != 1 or result[result.index('--num-layers') + 1] != '3':
        raise ValueError('Expected original three-layer trainer command')
    if result.count('--split-file') != 1:
        raise ValueError('Fixed image split required')
    result[result.index('--num-layers') + 1] = '2'
    return result


def worker(job):
    if job['stage'] not in ('train', 'eval') or job['seed'] not in SEEDS:
        raise ValueError('Only fixed-seed train/eval jobs are allowed')
    role = job['role']
    if role == 'smfa2':
        if job['stage'] != 'eval' or job['group'] != 'SMFA' or job['variant'] != 'd2':
            raise ValueError('SMFA2 is read-only')
        return mlp.worker(job)
    if role not in ('gmp3', 'gmp2') or job['group'] != 'C':
        raise ValueError('Only single-phase hard-center GMP C is allowed')
    expected = queue.signature({}, 'combo', queue.Recipe('C', job['checkpoint_sha256']))
    if (job['raw_signature']['format_version'] != expected['format_version']
            or job['raw_signature']['experimental_pooling'] != expected['experimental_pooling']):
        raise ValueError('Wrong aggregation/phase/rho/checkpoint signature')
    if role == 'gmp3' and job['stage'] != 'eval':
        raise ValueError('Historical GMP3 is read-only')
    if role == 'gmp2':
        mlp.install_adapter('d2')
        if job['stage'] == 'train':
            original = ab.runpy.run_path
            def train(path, *args, **kwargs):
                if Path(path).resolve() != ROOT / 'tools/train_feature_adapter.py':
                    raise ValueError('Unexpected trainer')
                # combo.worker has already substituted the requested seed.
                sys.argv = depth_two_argv(sys.argv)
                ab.write(Path(job['attempt']) / 'effective_training_argv.json', dict(
                    argv=sys.argv, seed=job['seed'], architecture=mlp.adapter_tag('d2')))
                return original(path, *args, **kwargs)
            ab.runpy.run_path = train
    return queue.worker(job)


def history(combo_folder, mlp_folder, clean_dir):
    common, smfa3, split, records, clean = mlp.verified_history(combo_folder, clean_dir)
    combo_manifest = ab.read(combo_folder / 'manifest.json')
    manifest = ab.read(mlp_folder / 'manifest.json')
    if (manifest['protocol'] != mlp.FORMAT or ab.read(mlp_folder / 'status.json')['status'] != 'complete'
            or Path(manifest['previous']).resolve() != combo_folder):
        raise ValueError('Need completed MLP experiment with the identical combo source')
    for field in ('code', 'references'):
        gmp.verify(manifest[field], 'MLP history ' + field)
        records += manifest[field]
    records += ab.identities([mlp_folder / n for n in ('manifest.json', 'status.json', 'summary.json')])
    raw = queue.verified_stage(combo_folder, 'merge_C', records)['raw']
    signature = ab.read(Path(raw) / 'meta/feature_signature.json')
    if signature != combo_manifest['signatures']['C']:
        raise ValueError('GMP C raw signature changed')
    expected = queue.signature(common['raw_signature'], 'combo', queue.Recipe('C', common['checkpoint_sha256']))
    if signature != expected:
        raise ValueError('Expected one-phase, uniform-token GMP rho=0.1')
    if ab.bank_members(raw) != ab.bank_members(smfa3[42]['raw']):
        raise ValueError('GMP and SMFA raw members differ')
    refs = dict(smfa3=smfa3, smfa2={}, gmp3={})
    for seed in SEEDS:
        for role, folder, key in [('gmp3', combo_folder, f'C_{seed}'), ('smfa2', mlp_folder, f'd2_{seed}')]:
            value = dict(raw=raw if role == 'gmp3' else smfa3[seed]['raw'])
            for stage in ('train', 'eval'):
                marker = folder / f'{stage}_{key}' / 'complete.json'
                value.update(queue.verified_stage(folder, f'{stage}_{key}', records))
                records += ab.identities([marker])
            training = ab.read(Path(value['weights']) / 'training_report.json')
            if any(training[k] != split[k] for k in split):
                raise ValueError('Historical training splits differ')
            if (training['args']['seed'] != seed or training['args']['num_layers'] != (3 if role == 'gmp3' else 2)
                    or Path(training['args']['source_bank']).resolve() != Path(value['raw']).resolve()):
                raise ValueError('Historical depth/seed/raw bank differs')
            bank_sig = ab.read(Path(value['bank']) / 'meta/feature_signature.json')
            underlying = dict(bank_sig)
            underlying.pop('adapter')
            underlying['embed_dim'] = 1280
            if underlying != (signature if role == 'gmp3' else common['raw_signature']):
                raise ValueError('Historical adapter source space differs')
            if role == 'smfa2' and bank_sig['adapter']['architecture'].get('experimental_mlp') != mlp.adapter_tag('d2'):
                raise ValueError('Historical SMFA2 is not the tested plain two-layer adapter')
            metric = ab.read(value['metrics'])
            mlp.validate_metrics(metric, ab.read(smfa3[seed]['metrics']), common['val_ids'], clean['count'])
            seeds.validate_image_matrices(value['per_image'], metric)
            refs[role][seed] = value
    import numpy as np
    for stem in common['val_ids']:
        with np.load(clean_dir / f'{stem}.npz', allow_pickle=False) as proposals:
            queue.cache_vectors(Path(common['cache']) / 'val' / stem, common['cache_identity'][stem],
                                'C', queue.mask_hashes(proposals['masks']))
    return common, refs, split, records, clean, raw, signature


def summarize(reports):
    roles = ('smfa3', 'smfa2', 'gmp3', 'gmp2')
    groups = {}
    for role in roles:
        values = [r['miou'][role] for r in reports.values()]
        groups[role] = dict(complete=len(values) == 3,
            mean_miou=statistics.mean(values) if values else None,
            sample_std_miou=statistics.stdev(values) if len(values) > 1 else None)
    contrasts = {}
    for name in ('gmp2-smfa2', 'gmp2-gmp3', 'gmp2-smfa3', 'interaction'):
        values = {seed: r['delta_pp'][name] for seed, r in reports.items()}
        contrasts[name] = dict(by_seed=values, mean_pp=statistics.mean(values.values()) if values else None,
            sample_std_pp=statistics.stdev(values.values()) if len(values) > 1 else None)
    complete = len(reports) == 3
    return dict(groups=groups, contrasts=contrasts,
        new_high_mean_vs_references=(groups['gmp2']['mean_miou'] > max(groups[r]['mean_miou'] for r in roles[:-1])) if complete else None,
        new_high_single_vs_references=(max(r['miou']['gmp2'] for r in reports.values()) >
            max(r['miou'][role] for r in reports.values() for role in roles[:-1])) if complete else None,
        note='Development experiment chosen after VOC val analysis; no statistical significance or automatic deployment.')


def compare(result, references, seed, ids, count):
    candidate = ab.read(result['metrics'])
    values = {role: ab.read(value['metrics']) for role, value in references.items()}
    for value in values.values():
        mlp.validate_metrics(candidate, value, ids, count)
    values['gmp2'] = candidate
    scores = {role: value['mIoU'] for role, value in values.items()}
    delta = {f'gmp2-{role}': 100 * (scores['gmp2'] - scores[role]) for role in references}
    delta['interaction'] = 100 * ((scores['gmp2'] - scores['gmp3']) - (scores['smfa2'] - scores['smfa3']))
    return dict(seed=seed, miou=scores, delta_pp=delta, evaluated_ids=ids,
        per_class_iou={role: value['per_class_iou'] for role, value in values.items()},
        confusion_matrices={role: value['confusion_matrix'] for role, value in values.items()},
        per_image_files={**{role: value['per_image'] for role, value in references.items()}, 'gmp2': result['per_image']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--combo-previous', type=Path, default=Path('experiments/mpsa_combo_20260929_105543'))
    parser.add_argument('--mlp-previous', type=Path, default=Path('experiments/smfa_mlp_20261001_221703'))
    parser.add_argument('--clean-proposals-dir', type=Path, default=Path(mpsa.DEFAULT_CLEAN))
    parser.add_argument('--version', default='gmp_mlp2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus', nargs=2, type=int, default=[0, 1])
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    opts = parser.parse_args(argv)
    os.chdir(ROOT)
    if opts.worker:
        return worker(ab.read(opts.worker))
    ab.parse_args(['--version', opts.version])
    if len(set(opts.gpus)) != 2 or min(opts.gpus) < 0 or not math.isfinite(opts.max_hours) or opts.max_hours <= 0:
        parser.error('Invalid GPUs/time budget')
    for gpu in opts.gpus:
        ab.check_gpu(gpu)
    combo_folder, mlp_folder = opts.combo_previous.resolve(), opts.mlp_previous.resolve()
    common, refs, split, records, clean, raw, signature = history(combo_folder, mlp_folder, opts.clean_proposals_dir.resolve())
    output, logs = ROOT / 'experiments' / opts.version, ROOT / 'logs' / opts.version
    weights, banks = ROOT / 'weights/sam3_adapter' / opts.version, ROOT / 'storage/mlp_banks/voc' / opts.version
    code = ab.source_identity()
    identity = dict(protocol=PROTOCOL, code=code, references=records, gpus=opts.gpus,
        seeds=list(SEEDS), combo_previous=str(combo_folder), mlp_previous=str(mlp_folder),
        raw_signature=signature, architecture=mlp.adapter_tag('d2'), clean=clean)
    identity = json.loads(json.dumps(identity))
    if output.exists():
        if not opts.resume or ab.read(output / 'manifest.json') != identity:
            raise ValueError('Existing version differs or --resume missing')
    elif any(p.exists() for p in (logs, weights, banks)):
        raise FileExistsError('Output version already used')
    members = sum(map(len, ab.bank_members(raw).values()))
    estimate = members * 256 * 4 * 3 + 2**30
    if mlp.shutil.disk_usage(ROOT).free < queue.FLOOR + (0 if opts.resume else estimate):
        raise queue.Deferred('Need estimated three exported banks + 1 GiB overhead + 5 GiB free reserve')
    stages = ab.Stages(output, True)
    stages.verify_all()
    print('[Plan] Hard-center single-phase GMP rho=.1 + plain 1280->512->256; 3 new trainings.', flush=True)
    print('[Plan] Replay GMP3/SMFA2 x3 first; reuse masks, raw bank and vectors; no extraction/cleaning.', flush=True)
    if opts.preflight_only:
        print('[Preflight passed]', flush=True)
        return
    if not output.exists():
        ab.write(output / 'manifest.json', identity)
    lock = output / 'launcher.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    budget = output / 'budget.json'
    if not budget.exists():
        ab.write(budget, dict(started_at=time.time()))
    deadline = ab.read(budget)['started_at'] + opts.max_hours * 3600
    cancel, mutex = threading.Event(), threading.Lock()
    saved_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def stop(*_):
        cancel.set()
        raise KeyboardInterrupt('Stopped by user')
    for sig in saved_signals:
        signal.signal(sig, stop)
    # Reuse the tested process supervisor, with this file as its worker entry.
    # This changes a Python module variable only, never the prior script on disk.
    old_entry = mlp.__file__
    mlp.__file__ = __file__
    reports = {p.parent.name.removeprefix('report_'): ab.read(p)['result'] for p in output.glob('report_*/complete.json')}
    status, error, failures = 'running', None, {}
    replay = {'smfa3': refs['smfa3'], 'smfa2': {}, 'gmp3': {}}

    def flush():
        with mutex:
            value = dict(status=status, error=error, failures=failures, results=reports, **summarize(reports))
            value['workers'] = {p.parent.name: ab.read(ab.read(p)['result']['queue_worker'])
                for p in output.glob('*/complete.json') if 'queue_worker' in ab.read(p)['result']}
            ab.write(output / 'summary.json', value)
            ab.write(output / 'status.json', dict(status=status, completed=len(reports), expected=3, error=error))
            rows = ['| Method | Mean mIoU % | SD pp |', '|---|---:|---:|']
            for role, group in value['groups'].items():
                fmt = lambda v: '—' if v is None else f'{100*v:.4f}'
                rows.append(f'| {role} | {fmt(group["mean_miou"])} | {fmt(group["sample_std_miou"])} |')
            (output / 'summary.md').write_text(f'Status: {status}; {len(reports)}/3 seeds.\n\n' + '\n'.join(rows)
                + '\n\nPartial summaries use only completed paired seeds. See summary.json for paired differences and interaction.\n', encoding='utf-8')

    def execute(key, stage, role, seed, gpu, source, accepted=False):
        def action(attempt):
            job = {k: common[k] for k in ('dataset_root', 'image_dir', 'checkpoint', 'train_list', 'val_list',
                'imagelevel_json', 'gt_dir', 'images', 'train_ids', 'val_ids', 'cache', 'cache_identity',
                'checkpoint_sha256', 'proposals')}
            group = 'SMFA' if role == 'smfa2' else 'C'
            job.update(stage=stage, role=role, group=group, variant='d2' if role != 'gmp3' else 'd3',
                seed=seed, attempt=str(attempt), raw_signature=common['raw_signature'] if group == 'SMFA' else signature)
            env = ab.base_env(gpu)
            env['OVERRIDE_SAM3_CKPT'] = common['checkpoint']
            artifacts, result = [attempt], {}
            if stage == 'train':
                w, b = weights / str(seed) / attempt.name, banks / str(seed) / attempt.name
                job.update(raw=raw, weights=str(w), bank=str(b),
                    split_file=str(Path(refs['smfa3'][42]['weights']) / 'training_report.json'))
                artifacts += [w, b]
                result.update(weights=str(w), bank=str(b))
            else:
                job.update(weights=source['weights'], bank=source['bank'], metrics=str(attempt / 'metrics.json'), cleanup='PRE_CLEANED')
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(source['weights']) / 'adapter.pt'),
                    OVERRIDE_INF_FEATURE_BANK_ROOT=source['bank'], OVERRIDE_INF_PROPOSALS_DIR=common['proposals'])
                result.update(metrics=job['metrics'], per_image=str(attempt / 'per_image_confusion.npz'))
            log = logs / f'{key}_{attempt.name}.log'
            mlp.launch(job, attempt, env, log, cancel, float('inf') if accepted else deadline)
            if ab.read(attempt / 'queue_worker.json')['status'] != 'complete':
                raise ValueError('Worker incomplete')
            if stage == 'train':
                training = ab.read(Path(result['weights']) / 'training_report.json')
                if any(training[k] != split[k] for k in split):
                    raise ValueError('Fixed split changed')
                effective = ab.read(attempt / 'effective_training_argv.json')
                if effective['seed'] != seed or effective['architecture'] != mlp.adapter_tag('d2'):
                    raise ValueError('Effective adapter architecture/seed mismatch')
                sig = ab.read(Path(result['bank']) / 'meta/feature_signature.json')
                if sig['adapter']['architecture']['experimental_mlp'] != mlp.adapter_tag('d2'):
                    raise ValueError('Exported adapter mismatch')
                if ab.bank_members(result['bank']) != ab.bank_members(raw):
                    raise ValueError('Export members differ')
            else:
                metric = ab.read(result['metrics'])
                seeds.validate_image_matrices(result['per_image'], metric)
                mlp.validate_metrics(metric, ab.read(refs['smfa3'][seed]['metrics']), common['val_ids'], clean['count'])
                if metric['feature_signature'] != ab.read(Path(source['bank']) / 'meta/feature_signature.json'):
                    raise ValueError('Evaluated signature differs')
            result['queue_worker'] = str(attempt / 'queue_worker.json')
            return result, artifacts + [log]
        try:
            return stages.run(key, action)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise queue.GlobalFailure(str(exc)) from exc

    def replay_task(item, gpu):
        role, seed = item
        result = dict(refs[role][seed])
        result.update(execute(f'replay_{role}_{seed}', 'eval', role, seed, gpu, result))
        mpsa.require_baseline_replay(ab.read(refs[role][seed]['metrics']), ab.read(result['metrics']))
        replay[role][seed] = result

    def task(seed, gpu):
        if str(seed) in reports:
            return
        result = dict(raw=raw)
        result.update(execute(f'train_{seed}', 'train', 'gmp2', seed, gpu, result, True))
        result.update(execute(f'eval_{seed}', 'eval', 'gmp2', seed, gpu, result, True))
        value = compare(result, {role: replay[role][seed] for role in replay}, seed, common['val_ids'], clean['count'])
        def action(attempt):
            path = output / f'comparison_seed{seed}.json'
            ab.write(path, value)
            return value, [path]
        saved = stages.run(f'report_{seed}', action)
        with mutex:
            reports[str(seed)] = saved
        flush()

    def fail(seed, exc):
        with mutex:
            failures[str(seed)] = str(exc)
        flush()

    try:
        ab.write(output / 'artifacts.json', dict(owned_roots=list(map(str, (output, logs, weights, banks))),
            read_only_sources=[str(combo_folder), str(mlp_folder), raw, common['cache'], common['proposals']], automatic_deletion=False))
        flush()
        queue.run_round([(role, seed) for role in ('smfa2', 'gmp3') for seed in SEEDS], opts.gpus,
            replay_task, cancel, deadline, lambda item, exc: (_ for _ in ()).throw(queue.GlobalFailure(f'Replay failed: {exc}')))
        print('[Replay passed] Six historical full confusion matrices reproduced.', flush=True)
        queue.run_round(list(SEEDS), opts.gpus, task, cancel, deadline, fail)
        stages.verify_all()
        gmp.verify(records, 'Historical inputs/artifacts')
        if ab.source_identity() != code:
            raise queue.GlobalFailure('Code changed during run')
        status = 'complete' if len(reports) == 3 else 'complete_with_failures'
    except queue.Deferred as exc:
        status, error = 'deferred', str(exc)
    except BaseException as exc:
        cancel.set()
        status, error = 'incomplete', str(exc)
        raise
    finally:
        try:
            flush()
        finally:
            mlp.__file__ = old_entry
            for sig, handler in saved_signals.items():
                signal.signal(sig, handler)
            lock.unlink()
    print(f'[{status}] {output}/summary.md', flush=True)


if __name__ == '__main__':
    main()
