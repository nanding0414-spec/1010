#!/usr/bin/env python3
"""Disposable RPG-SMFA ablations with a verified, read-only SMFA baseline."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_pmfa_ab as ab


@dataclass(frozen=True)
class RPG:
    variant: str = 'D'
    lambda_base: float = .5
    temperature: float = .1
    core_threshold: float = .8
    core_ratio: float = .3

    def __post_init__(self):
        if self.variant not in ('B', 'C', 'D'):
            raise ValueError('variant must be B, C or D')
        for key in ('lambda_base', 'temperature', 'core_threshold', 'core_ratio'):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'{key} must be finite')
        if not 0 <= self.lambda_base <= 1 or not .5 <= self.core_threshold <= 1:
            raise ValueError('lambda_base must be in [0,1], core_threshold in [0.5,1]')
        if self.temperature <= 0 or not 0 < self.core_ratio <= 1:
            raise ValueError('temperature must be positive and core_ratio in (0,1]')


def signature(base, mode, params):
    if mode != 'rpg':
        raise ValueError('RPG worker only accepts rpg mode')
    return dict(base, format_version=3002, experimental_pooling=dict(
        version=1, mode='rpg', parameters=asdict(params), precision_policy=ab.POLICY,
        q0='normalize_coverage_weighted_raw_tokens', median='torch_lower_median',
        placement='before_region_mlp', mask_chunk_size=16, coverage_tolerance=1e-6,
        effective_lambda=params.lambda_base if params.variant == 'D' else 0.))


def pool(tokens, weights, params, original, stats=None, diagnostics=None):
    """Detached gate, differentiable value aggregation, original final L2 norm."""
    import torch
    import torch.nn.functional as F
    if tokens.ndim != 2 or weights.ndim != 1 or len(tokens) != len(weights):
        raise ValueError('Expected tokens[N,D], coverage[N]')
    if not tokens.is_floating_point() or not weights.is_floating_point():
        raise ValueError('Tokens and coverage must be floating point')
    if tokens.device != weights.device:
        raise ValueError('Tokens and coverage must share a device')
    if not torch.isfinite(tokens).all() or not torch.isfinite(weights).all():
        raise ValueError('Non-finite tokens or coverage')
    if ((weights < -1e-6) | (weights > 1+1e-6)).any():
        raise ValueError(f'Coverage outside tolerance: {float(weights.min())}, {float(weights.max())}')
    stats = stats if stats is not None else Counter()
    stats['regions'] += 1
    m = weights.detach().float().clamp(0, 1)
    valid = m > 0
    n = int(valid.sum())
    record = dict(valid_tokens=n, core_tokens=0, selected_tokens=0, fallback=None)
    reference = original(tokens, weights)

    def finish(value, reason=None):
        if reason:
            record['fallback'] = reason
            stats['fallback_'+reason] += 1
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite RPG output')
        record['output_smfa_cosine'] = float(F.cosine_similarity(
            value.detach().float()[None], reference.detach().float()[None], eps=1e-6))
        if diagnostics is not None:
            diagnostics(record)
        return value

    stats['coverage_roundoff_tokens'] += int(((weights < 0) | (weights > 1)).sum())
    if params.variant == 'D' and params.lambda_base == 1:
        return finish(reference, 'lambda_one')
    if n < 3 or float(m.sum()) < 1e-6:
        return finish(reference, 'small_or_empty')
    v = tokens[valid].float()
    m = m[valid]
    with torch.no_grad():
        vn = F.normalize(v.detach(), dim=-1, eps=1e-6)
        q0 = (v.detach()*m[:, None]).sum(0)/(m.sum()+1e-8)
        if q0.norm() < 1e-6:
            return finish(reference, 'zero_q0')
        q0 = F.normalize(q0, dim=0, eps=1e-6)
        q1 = q0
        if params.variant != 'B':
            core = m >= params.core_threshold
            if int(core.sum()) < 3:
                core = m >= .5
                stats['core_threshold_half'] += 1
            if int(core.sum()) < 3:
                core = torch.ones_like(m, dtype=torch.bool)
                stats['core_all_valid'] += 1
            indices = torch.where(core)[0]
            k = min(len(indices), max(3, math.ceil(params.core_ratio*len(indices))))
            selected = indices[torch.topk(vn[indices]@q0, k, sorted=False).indices]
            record.update(core_tokens=len(indices), selected_tokens=k)
            q1 = (vn[selected]*m[selected, None]).sum(0)/(m[selected].sum()+1e-8)
            if q1.norm() < 1e-6:
                return finish(reference, 'zero_q1')
            q1 = F.normalize(q1, dim=0, eps=1e-6)
        sim = vn@q1
        gate = torch.sigmoid((sim-sim.median())/params.temperature)
        floor = params.lambda_base if params.variant == 'D' else 0.
        factors = floor+(1-floor)*gate
        w = m*factors
        record.update(proxy_cosine=float(q0@q1), gate_mean=float(gate.mean()),
                      gate_min=float(gate.min()), gate_max=float(gate.max()),
                      factor_min=float(factors.min()), factor_max=float(factors.max()))
        if float(w.sum()) < 1e-6:
            return finish(reference, 'zero_weight')
    stats['proxy_regions'] += 1
    result = F.normalize((v*w[:, None]).sum(0)/(w.sum()+1e-8), dim=0)
    return finish(result)


def install_pooling(mode, params, stats, job):
    import json
    import numpy as np
    import torch
    from src.feature_extractor import BaseFeatureExtractor as Base
    from src.feature_extractor_sam3 import SAM3FeatureExtractor as Sam
    import src.feature_signature as fs
    fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
    old_sig = Sam.feature_signature
    Sam.feature_signature = lambda self: signature(old_sig(self), mode, params)
    old_pool = Base._pool_tokens
    image_state = dict(id=None, region=0)
    diagnostic_file = Path(job['attempt'])/'region_diagnostics.jsonl'

    def timed_image(call):
        started=time.perf_counter();before=stats['aggregation_seconds']
        try:
            return call()
        finally:
            row=dict(image_id=image_state['id'],regions=image_state['region'],
                     processing_seconds=time.perf_counter()-started,
                     aggregation_seconds=stats['aggregation_seconds']-before)
            with (Path(job['attempt'])/'image_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')

    def record(value):
        row = dict(image_id=image_state['id'], region_index=image_state['region'], **value)
        image_state['region'] += 1
        with diagnostic_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def one(self, tokens, mask):
        if self.context_subtract != 0:
            raise ValueError('RPG requires context_subtract=0')
        started = time.perf_counter()
        def diagnostic(value):
            value['aggregation_seconds'] = time.perf_counter()-started
            stats['aggregation_seconds'] += value['aggregation_seconds']
            record(value)
        return pool(tokens, torch.as_tensor(mask, device=tokens.device, dtype=torch.float32), params,
                    lambda t,w: old_pool(self,t,w.detach().cpu().numpy()), stats, diagnostic)

    def batch(self, rgb, masks):
        if len(masks) == 0:
            return []
        inp, transform = self._preprocess_image(rgb)
        tokens, (hp, wp) = self._forward_tokens(inp)
        if tokens is None or tokens.shape[0] != 1:
            raise ValueError('Expected spatial tokens for one image')
        output = []
        for start in range(0, len(masks), 16):
            for mask in masks[start:start+16]:
                coverage = self._downsample_mask_to_patches(np.asarray(mask,dtype=bool),hp,wp,transform)
                output.append(one(self,tokens[0],coverage).detach().cpu().numpy())
        return output

    old_regions = Base.extract_features_for_regions
    def regions(self, path, *args, **kwargs):
        image_state.update(id=Path(path).stem, region=0)
        try:
            return timed_image(lambda: old_regions(self,path,*args,**kwargs))
        finally:
            image_state.update(id=None, region=0)
    Base._pool_tokens, Base.extract_features_batch = one, batch
    Base.extract_features_for_regions = regions
    if job['stage']=='eval':
        # Preserve the worker's dataset override before binding inference imports.
        import configs.config as cfg
        cfg.DATASET_ROOT=cfg.INF_DATASET_ROOT=job['dataset_root']
        import run_inference as inference
        inference.load_eval_samples=lambda _: [dict(id=i,img=job['images'][i],
            gt=str(Path(job['gt_dir'])/f'{i}.png')) for i in job['val_ids']]
        old_load=inference.load_strict_proposals
        def load(directory,image_id,height,width):
            image_state.update(id=image_id,region=0)
            return old_load(directory,image_id,height,width)
        inference.load_strict_proposals=load
        old_predict=inference.predict_instances
        inference.predict_instances=lambda *a,**kw: timed_image(lambda:old_predict(*a,**kw))
    print(f'[RPG] variant={params.variant}; aggregation before MLP; params={asdict(params)}',flush=True)


def worker(job):
    ab.PMFA = RPG
    ab.EXPERIMENT_FORMAT = 3002
    ab.signature = signature
    ab.install_pooling = lambda mode,params,stats: install_pooling(mode,params,stats,job)
    ab.worker(job)


def verify_records(records, label):
    expected = {}
    for row in records:
        path = row['path']
        if path in expected and expected[path] != row:
            raise ValueError(f'{label}: conflicting records for {path}')
        expected[path] = row
    try:
        actual = {row['path']: row for row in ab.identities(expected)}
    except OSError as exc:
        raise ValueError(f'{label} files changed or missing: {exc}') from exc
    changed = [path for path, row in expected.items() if actual.get(path) != row]
    if changed:
        raise ValueError(f'{label} files changed or missing: {changed[:10]}')


def baseline(folder):
    print(f'[Baseline] checking source, inputs and completed artifacts: {folder}',flush=True)
    manifest = ab.read(folder/'manifest.json')
    if ab.read(folder/'status.json')['status'] != 'complete':
        raise ValueError('Baseline run is not complete')
    if ab.read(folder/'comparison.json')['status'] != 'complete':
        raise ValueError('Baseline comparison is not complete')
    if manifest['precision_policy'] != ab.POLICY:
        raise ValueError('Baseline precision differs')
    verify_records(manifest['inputs'], 'Baseline inputs/weights')
    verify_records(manifest['code'], 'Baseline recorded source')
    # Added experiment tools are permitted; newly added production code is not.
    old_paths = {r['path'] for r in manifest['code']}
    for row in ab.source_identity():
        if row['path'] not in old_paths:
            path = Path(row['path'])
            if path.parent != ROOT/'tools' or not path.name.startswith('experimental_'):
                raise ValueError(f'New unverified production source: {path}')
    wanted = {k:v for k,v in ab.base_env(1).items() if k.startswith('OVERRIDE_')}
    got = {k:v for k,v in manifest['environment'].items() if k.startswith('OVERRIDE_')}
    if wanted != got:
        raise ValueError('Baseline execution recipe differs')
    records = []; result = {}
    for kind in ('pseudo','proposals'):
        if not manifest['selected'].get(kind):
            marker=ab.read(folder/kind/'complete.json')
            verify_records(marker['artifacts'],f'Baseline {kind}')
            records += marker['artifacts']
    for kind in ('build','train','eval'):
        marker = ab.read(folder/f'smfa_{kind}'/'complete.json')
        verify_records(marker['artifacts'], f'Baseline {kind}')
        records += marker['artifacts']
        result.update({k:v for k,v in marker['result'].items() if k != 'worker_report'})
    sig = ab.read(Path(result['raw'])/'meta/feature_signature.json')
    expected = ab.signature({}, 'smfa', ab.PMFA())['experimental_pooling']
    if sig['experimental_pooling'] != expected or sig['format_version'] != 3001:
        raise ValueError('Baseline is not the supported experimental SMFA signature')
    records += ab.identities([folder/'manifest.json',folder/'status.json',folder/'comparison.json'])
    return manifest, result, records


def comparison(base, current, context, params):
    import numpy as np
    a,b = [ab.read(x['metrics']) for x in (base,current)]
    for value in (a,b):
        if (value['evaluated_ids'] != context['val_ids'] or value['evaluated_count'] != len(context['val_ids'])
            or value['eval_limit'] != 0 or not math.isfinite(value['mIoU']) or not 0<=value['mIoU']<=1):
            raise ValueError('Incomplete or invalid evaluation')
    if list(a['per_class_iou']) != list(b['per_class_iou']): raise ValueError('Class order differs')
    matrices=[np.asarray(v['confusion_matrix']) for v in (a,b)]
    for value,matrix in zip((a,b),matrices):
        if matrix.shape!=(21,21) or not np.isfinite(matrix).all() or (matrix<0).any():
            raise ValueError('Invalid confusion matrix')
        iou=np.diag(matrix)/(matrix.sum(0)+matrix.sum(1)-np.diag(matrix)).clip(min=1)
        if not np.allclose(iou,list(value['per_class_iou'].values()),atol=2e-6,rtol=0):
            raise ValueError('Confusion matrix and class IoUs disagree')
        if abs(float(iou.mean())-value['mIoU'])>2e-6: raise ValueError('Confusion matrix and mIoU disagree')
    if not np.array_equal(matrices[0].sum(1),matrices[1].sum(1)):
        raise ValueError('Ground truth pixel counts differ')
    for key in ('cleanup','proposals_dir','proposal_count_before','proposal_count_after'):
        if a[key] != b[key]: raise ValueError(f'Comparison mismatch: {key}')
    ignore = {'INF_FEATURE_BANK_ROOT','INF_SAVE_DIR'}
    if ({k:v for k,v in a['inference_settings'].items() if k not in ignore} !=
        {k:v for k,v in b['inference_settings'].items() if k not in ignore}):
        raise ValueError('Inference settings differ')
    for result,metrics,method in ((base,a,'smfa'),(current,b,'rpg')):
        sig = ab.read(Path(result['bank'])/'meta/feature_signature.json')
        if sig != metrics['feature_signature'] or sig['experimental_pooling']['mode'] != method:
            raise ValueError('Evaluated bank signature mismatch')
    if b['feature_signature']['experimental_pooling'] != signature({},'rpg',params)['experimental_pooling']:
        raise ValueError('RPG parameters differ')
    splits = [ab.read(Path(x['weights'])/'training_report.json') for x in (base,current)]
    if splits[0]['validation_images'] != splits[1]['validation_images']:
        raise ValueError('Internal validation membership differs')
    members = [ab.bank_members(x['raw']) for x in (base,current)]
    counts = {c:dict(smfa=len(members[0].get(c,set())),rpg=len(members[1].get(c,set())),
        common=len(members[0].get(c,set())&members[1].get(c,set())),
        smfa_only=sorted(members[0].get(c,set())-members[1].get(c,set())),
        rpg_only=sorted(members[1].get(c,set())-members[0].get(c,set())))
        for c in sorted(set(members[0])|set(members[1]))}
    delta = 100*(b['mIoU']-a['mIoU'])
    per_class = {c:dict(smfa=v,rpg=b['per_class_iou'][c],delta_pp=100*(b['per_class_iou'][c]-v))
                 for c,v in a['per_class_iou'].items()}
    return dict(status='complete',variant=params.variant,parameters=asdict(params),smfa_miou=a['mIoU'],
        rpg_miou=b['mIoU'],delta_pp=delta,outcome='提高' if delta>1e-8 else '下降' if delta< -1e-8 else '持平',
        baseline_reused=True,note='Single seed; separate filtering and newly trained RPG MLP; no significance claim.',
        per_class_iou=per_class,background=per_class['background'],bank_membership=counts,
        evaluated_ids=context['val_ids'],confusion_matrices=dict(smfa=a['confusion_matrix'],rpg=b['confusion_matrix']),
        internal_split_coverage={m:{k:s[k] for k in ('train_images','validation_images')}
                                 for m,s in zip(('smfa','rpg'),splits)})


def run_variant(version, gpu, params, args, context, base, baseline_records, cancel):
    output = ROOT/'experiments'/version
    roots = dict(raw=ROOT/'storage/raw_banks/voc'/version,bank=ROOT/'storage/mlp_banks/voc'/version,
                 weights=ROOT/'weights/sam3_adapter'/version,logs=ROOT/'logs'/version)
    identity = dict(parameters=asdict(params),gpu=gpu,previous=str(args.previous),inputs=context['inputs'],
                    code=context['code'],baseline_artifacts=baseline_records,precision_policy=ab.POLICY)
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json') != identity:
            raise ValueError(f'Existing version differs or needs --resume: {version}')
    else:
        if any(p.exists() for p in roots.values()): raise FileExistsError(version)
        ab.write(output/'manifest.json',identity)
    lock = output/'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    stages = ab.Stages(output,args.resume)
    result = {}; common = dict(pmfa=asdict(params),mode='rpg',train_ids=context['train_ids'],
        val_ids=context['val_ids'],images=context['images'])
    for key in ('dataset_root','image_dir','imagelevel_json','checkpoint','train_list','val_list','gt_dir'):
        common[key] = str(getattr(args,key))
    env = ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT'] = str(args.checkpoint)
    from configs.experiment import CLEANUP
    try:
        ab.write(output/'artifacts.json',dict(owned_roots=[str(output),*[str(p) for p in roots.values()]],
                  read_only_inputs=context['selected'],read_only_baseline=str(args.previous),automatic_deletion=False))
        ab.write(output/'cleanup.json',CLEANUP)
        for kind in ('probe','build','train','eval'):
            if cancel.is_set(): raise RuntimeError('Experiment cancelled')
            def action(attempt):
                job = dict(common,stage=kind,attempt=str(attempt)); current=dict(env); artifacts=[attempt]; values={}
                if kind == 'build':
                    raw=roots['raw']/attempt.name
                    current.update(OVERRIDE_OUTPUT_ROOT=str(raw),OVERRIDE_PSEUDO_MASK_ROOT=context['selected']['pseudo'],
                                   OVERRIDE_BUILD_IMAGE_DIR=str(args.image_dir))
                    job['pseudo']=context['selected']['pseudo'];values['raw']=str(raw);artifacts.append(raw)
                elif kind == 'train':
                    weights=roots['weights']/attempt.name;bank=roots['bank']/attempt.name
                    job.update(raw=result['raw'],weights=str(weights),bank=str(bank),
                               split_file=str(Path(base['weights'])/'training_report.json'))
                    values.update(weights=str(weights),bank=str(bank));artifacts += [weights,bank]
                elif kind == 'eval':
                    job.update(metrics=str(attempt/'metrics.json'),cleanup=str(output/'cleanup.json'))
                    current.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(result['weights'])/'adapter.pt'),
                        OVERRIDE_INF_FEATURE_BANK_ROOT=result['bank'],OVERRIDE_INF_PROPOSALS_DIR=context['selected']['proposals'])
                    values['metrics']=job['metrics']
                log=roots['logs']/f'{kind}_{attempt.name}.log'
                ab.launch(job,attempt,current,log,cancel=cancel)
                if ab.read(attempt/'worker_report.json')['status'] != 'complete': raise RuntimeError('Worker incomplete')
                values['worker_report']=str(attempt/'worker_report.json')
                return values, artifacts+[log]
            result.update(stages.run(kind,action))
            ab.write(output/'status.json',dict(status='in_progress',completed_stage=kind))
        stages.verify_all();verify_records(baseline_records,'Baseline artifacts')
        verify_records(context['inputs'],'Experiment inputs')
        if ab.source_identity() != context['code']: raise ValueError('Source changed during experiment')
        report=comparison(base,result,context,params)
        report['baseline_source']=dict(experiment=str(args.previous),metrics=base['metrics'],
                                      training_report=str(Path(base['weights'])/'training_report.json'))
        report['stages']={p.parent.name:ab.read(ab.read(p)['result']['worker_report']) for p in output.glob('*/complete.json')}
        ab.write(output/'comparison.json',report)
        (output/'comparison.md').write_text(
            f'# RPG {params.variant} / SMFA\n\nSMFA (verified reuse): {100*report["smfa_miou"]:.4f}%\n\n'
            f'RPG: {100*report["rpg_miou"]:.4f}%\n\nDelta: {report["delta_pp"]:+.4f} pp\n\n'
            'Single seed; separate filtering; no statistical significance claim.\n',encoding='utf-8')
        (output/'per_class.csv').write_text('class,smfa_iou,rpg_iou,delta_pp\n'+''.join(
            f'{c},{v["smfa"]},{v["rpg"]},{v["delta_pp"]}\n' for c,v in report['per_class_iou'].items()),encoding='utf-8')
        ab.write(output/'status.json',dict(status='complete',result=result))
        print(f'[DONE] {version}: delta={report["delta_pp"]:+.4f} pp',flush=True)
        return report
    except BaseException as exc:
        ab.write(output/'status.json',dict(status='incomplete',error=str(exc),result=result));raise
    finally:
        lock.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--version',default='rpg_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    parser.add_argument('--variants',nargs='+',choices=['B','C','D'],default=['B','C','D'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    for key,value in asdict(RPG()).items():
        if key != 'variant': parser.add_argument('--'+key.replace('_','-'),type=float,default=value)
    opts=parser.parse_args(argv)
    os.chdir(ROOT)
    if opts.worker: return worker(ab.read(opts.worker))
    if opts.previous is None: parser.error('--previous is required: previous complete PMFA/SMFA experiment directory')
    if not opts.gpus or min(opts.gpus)<0 or len(set(opts.gpus))!=len(opts.gpus) or len(set(opts.variants))!=len(opts.variants):
        parser.error('GPU IDs and variants must be distinct; GPU IDs must be nonnegative')
    params={v:RPG(variant=v,**{k:getattr(opts,k) for k in asdict(RPG()) if k!='variant'}) for v in opts.variants}
    opts.previous=opts.previous.resolve()
    manifest,base,records=baseline(opts.previous)
    flags=['--version',opts.version,'--gpu',str(opts.gpus[0])]
    for key in ('dataset_root','image_dir','checkpoint','train_list','val_list','imagelevel_json','gt_dir'):
        flags += ['--'+key.replace('_','-'),manifest['options'][key]]
    for kind,flag in (('pseudo','--pseudo-dir'),('proposals','--proposals-dir')):
        directory=manifest['selected'].get(kind) or ab.read(opts.previous/kind/'complete.json')['result'][kind]
        flags += [flag,directory]
    args=ab.parse_args(flags);args.previous=opts.previous;args.resume=opts.resume
    context=ab.preflight(args)
    if not all(context['selected'].values()): raise ValueError('Shared caches invalid; refusing to regenerate')
    for gpu in opts.gpus[1:]: ab.check_gpu(gpu)
    print(f'[Baseline] reuse {opts.previous}; no SMFA workers will run',flush=True)
    if opts.preflight_only: return
    control=ROOT/'experiments'/opts.version
    plan=dict(previous=str(opts.previous),gpus=opts.gpus,variants=opts.variants,parameters={k:asdict(v) for k,v in params.items()})
    if control.exists():
        if not opts.resume or ab.read(control/'plan.json')!=plan: raise FileExistsError(control)
    else: ab.write(control/'plan.json',plan)
    lock=control/'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    cancel=threading.Event();old=signal.getsignal(signal.SIGTERM)
    def stop(*_):
        cancel.set();raise KeyboardInterrupt('RPG launcher interrupted')
    signal.signal(signal.SIGTERM,stop)
    ab.SCRIPT=Path(__file__).resolve()
    reports={}
    def lane(index,gpu):
        for variant in opts.variants[index::len(opts.gpus)]:
            if cancel.is_set(): return
            reports[variant]=run_variant(opts.version+'_'+variant,gpu,params[variant],args,context,base,records,cancel)
    try:
        with ThreadPoolExecutor(max_workers=len(opts.gpus)) as executor:
            try:
                pending={executor.submit(lane,i,g) for i,g in enumerate(opts.gpus)}
                while pending:
                    done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done: future.result()
            finally: cancel.set()
        ab.write(control/'summary.json',dict(status='complete',baseline=str(opts.previous),
            results={k:{x:r[x] for x in ('smfa_miou','rpg_miou','delta_pp','outcome')} for k,r in reports.items()}))
    except BaseException as exc:
        ab.write(control/'status.json',dict(status='incomplete',error=str(exc),completed_variants=sorted(reports)));raise
    finally:
        signal.signal(signal.SIGTERM,old);lock.unlink()


if __name__=='__main__':
    main()
