#!/usr/bin/env python3
"""Isolated coverage-weighted GMP: fixed members, two candidates, internal selection."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_pmfa_ab as ab
from tools import experimental_rpg_ab as rpg


@dataclass(frozen=True)
class GMP:
    rho: float = 1.0
    variant: str = 'gmp'
    def __post_init__(self):
        if not math.isfinite(self.rho) or self.rho<=0 or self.variant!='gmp':
            raise ValueError('GMP requires finite positive rho and variant=gmp')


def verify(records,label):
    expected={}
    for row in records:
        if row['path'] in expected and row!=expected[row['path']]: raise ValueError('Conflicting file identities')
        expected[row['path']]=row
    actual={r['path']:r for r in ab.identities(expected)}
    bad=[p for p,r in expected.items() if actual.get(p)!=r]
    if bad: raise ValueError(f'{label} changed: {bad[:5]}')


def signature(base,mode,params):
    if mode!='gmp': raise ValueError(mode)
    return dict(base,format_version=3004,experimental_pooling=dict(mode='gmp',version=1,
        parameters=asdict(params),precision_policy=ab.POLICY,coverage='p=m/sum(m)',
        ridge='rho*sum(p*norm(token)^2)',tokens='unchanged_processed_tokens',
        solver='fp32_cholesky_smaller_primal_or_dual',residual_limit=1e-4,
        final_normalization='l2',coverage_tolerance=1e-6,placement='before_region_mlp',
        empty_zero='zero',failure='raise_no_fallback',mask_chunk_size=16))


def solve(tokens,coverage,rho):
    import torch
    import torch.nn.functional as F
    GMP(rho)
    if tokens.ndim!=2 or coverage.shape!=(len(tokens),) or tokens.shape[1]<1:
        raise ValueError('Expected [N,C] and [N]')
    if not tokens.is_floating_point() or not coverage.is_floating_point() or tokens.device!=coverage.device:
        raise ValueError('Expected floating tensors on same device')
    if not torch.isfinite(tokens).all() or not torch.isfinite(coverage).all(): raise ValueError('Nonfinite input')
    if ((coverage< -1e-6)|(coverage>1+1e-6)).any(): raise ValueError('Coverage outside tolerance')
    m=coverage.float().clamp(0,1);valid=m>0;f=tokens[valid].float();m=m[valid]
    info=dict(support=len(m),rho=rho,roundoff_tokens=int(((coverage<0)|(coverage>1)).sum()))
    if len(m)==0: return tokens.new_zeros(tokens.shape[1],dtype=torch.float32),dict(info,zero='empty')
    p=m/m.sum();energy=(p*f.square().sum(1)).sum()
    if float(energy)==0: return tokens.new_zeros(tokens.shape[1],dtype=torch.float32),dict(info,zero='features')
    ridge=rho*energy;mu=p@f;a=p.sqrt()[:,None]*f
    primal=f.shape[1]<=len(f)
    matrix=a.T@a if primal else a@a.T
    matrix.diagonal().add_(ridge)
    chol,flag=torch.linalg.cholesky_ex(matrix)
    if int(flag)!=0: raise ValueError('GMP Cholesky failed; no silent solver/precision fallback')
    rhs=mu if primal else p.sqrt()
    x=torch.cholesky_solve(rhs[:,None],chol).squeeze(1)
    z=x if primal else a.T@x
    residual=(f.T@(p*(f@z))+ridge*z-mu).norm()/mu.norm().clamp_min(1e-12)
    if not torch.isfinite(z).all() or not torch.isfinite(residual) or float(residual)>1e-4:
        raise ValueError(f'GMP normal-equation relative residual: {float(residual)}')
    coefficients=p*(1-f@z)/ridge
    reference=F.normalize(mu,dim=0)
    output=F.normalize(z,dim=0)
    info.update(solver='primal' if primal else 'dual',ridge=float(ridge),energy=float(energy),
        relative_residual=float(residual),solution_norm=float(z.norm()),
        negative_coefficient_fraction=float((coefficients<0).float().mean()),
        low_coverage_absolute_coefficient_fraction=float(coefficients[m<.5].abs().sum()/coefficients.abs().sum().clamp_min(1e-12)),
        smfa_cosine=float(output@reference))
    return output,info


def pool(tokens,weights,params,original,stats=None,diagnostics=None):
    result,info=solve(tokens,weights,params.rho)
    if stats is not None:
        stats['regions']+=1;stats[info.get('solver','zero')]+=1
    if diagnostics is not None: diagnostics(info)
    return result


def read_bank(root):
    from src.feature_bank_storage import load_feature_matrix_npz
    records={}
    for file in sorted((Path(root)/'features').glob('*/*.npz')):
        features,ids=load_feature_matrix_npz(file)
        for value,sid in zip(features,ids):
            key=(file.parent.name,str(sid))
            if key in records: raise ValueError(f'Duplicate baseline region {key}')
            records[key]=value
    if not records: raise ValueError('Empty baseline bank')
    return records


def enumerate_members(train_ids,images,pseudo,raw):
    """Recreate original per-class global counters BEFORE filtering."""
    from src.region_extractors import get_regions_from_pseudo
    from configs.config import DATASET_CLASSES
    wanted=read_bank(raw);counter=Counter();found=set();plan={}
    coverage=ab.read(Path(raw)/'meta/coverage.json')['region_counts']
    for stem in train_ids:
        regions=get_regions_from_pseudo(images[stem],300,pseudo)
        if len(regions)!=coverage[stem]: raise ValueError(f'Region count changed: {stem}')
        selected=[]
        for index,region in enumerate(regions):
            cls='background' if region['cls_id']==0 else DATASET_CLASSES[region['cls_id']-1]
            sid=f'{stem}_{counter[cls]:06d}';counter[cls]+=1
            if (cls,sid) in wanted:
                found.add((cls,sid))
                selected.append(dict(index=index,cls=cls,id=sid,
                    mask_sha256=hashlib.sha256(region['mask'].tobytes()).hexdigest()))
        plan[stem]=selected
    if found!=set(wanted): raise ValueError('Cannot reconstruct baseline region membership')
    return plan


def image_cache(folder):
    marker=Path(folder)/'complete.json'
    if not marker.exists(): return None
    value=ab.read(marker);verify(value['artifacts'],'Image cache');return value['result']


def extract_worker(job):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0,str(ROOT/'sam3-main'))
    ab.install_precision()
    import configs.config as cfg
    cfg.DATASET_ROOT=cfg.INF_DATASET_ROOT=job['dataset_root']
    from src.feature_extractor_factory import build_feature_extractor
    from src.region_extractors import get_regions_from_pseudo
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.manual_seed(42)
    started=time.monotonic();model=build_feature_extractor(torch.device('cuda:0'),use_interactivity=False)
    reference=read_bank(job['baseline']['raw']);plan=ab.read(job['plan'])
    source_sig=model.feature_signature()
    expected=ab.read(Path(job['baseline']['raw'])/'meta/feature_signature.json')
    for key in source_sig:
        if key!='format_version' and source_sig[key]!=expected.get(key): raise ValueError(f'Base feature mismatch: {key}')
    for stem in job['ids']:
        folder=Path(job['cache'])/stem
        cached=image_cache(folder)
        if cached:
            if cached['source_signature']!=source_sig: raise ValueError('Cached feature signature changed')
            print(f'[Image resume] {stem}',flush=True);continue
        rows=plan[stem];regions=get_regions_from_pseudo(job['images'][stem],300,job['pseudo'])
        folder.mkdir(parents=True,exist_ok=True)
        number=1
        while (folder/f'attempt_{number:03d}').exists():number+=1
        attempt=folder/f'attempt_{number:03d}';attempt.mkdir()
        outputs=[[],[]];diagnostics=[];image_started=time.monotonic()
        with torch.inference_mode():
            if rows:
                with Image.open(job['images'][stem]) as im:rgb=np.array(im.convert('RGB'))
                inp,transform=model._preprocess_image(rgb);tokens,(hp,wp)=model._forward_tokens(inp)
                if tokens is None or tokens.shape[0]!=1:raise ValueError('Expected one-image spatial tokens')
                tok=tokens[0]
                for row in rows:
                    region=regions[row['index']]
                    if hashlib.sha256(region['mask'].tobytes()).hexdigest()!=row['mask_sha256']:
                        raise ValueError('Region mask changed')
                    mask=model._downsample_mask_to_patches(region['mask'].astype(bool),hp,wp,transform)
                    smfa=model._pool_tokens(tok,mask).cpu().numpy()
                    if not np.allclose(smfa,reference[(row['cls'],row['id'])],atol=5e-5,rtol=5e-5):
                        raise ValueError(f'Baseline region feature does not match reconstructed identity: {row["id"]}')
                    w=torch.as_tensor(mask,device=tok.device,dtype=torch.float32)
                    for slot,rho in enumerate((1.,.1)):
                        value,info=solve(tok,w,rho)
                        if float(value.norm())<1e-6:raise ValueError(f'Zero nonempty region: {stem}; no fallback')
                        outputs[slot].append(value.cpu().numpy());diagnostics.append(dict(sample_id=row['id'],cls=row['cls'],**info))
                    if job['stage']=='probe' and row is rows[0]:
                        # Diagnostic basis sampled only from this training region; no semantic claim.
                        valid=w>0;f=tok[valid].float();m=w[valid];count=min(512,len(f))
                        idx=torch.linspace(0,len(f)-1,count,device=f.device).long()
                        sample=f[idx];p=m[idx]/m[idx].sum()
                        for centered in (False,True):
                            origin=p@sample if centered else torch.zeros(f.shape[1],device=f.device)
                            _,_,vt=torch.linalg.svd(p.sqrt()[:,None]*(sample-origin),full_matrices=False)
                            for rank in (8,16):
                                basis=vt[:min(rank,len(vt))];x=f-origin
                                residual=(x-(x@basis.T)@basis).square().sum(1)
                                diagnostics.append(dict(diagnostic='sampled_basis_squared_residual',sample_id=row['id'],
                                    centered=centered,requested_rank=rank,actual_rank=len(basis),basis_sample_count=count,
                                    mean=float(residual.mean()),max=float(residual.max()),
                                    top_grid_indices=torch.where(valid)[0][torch.topk(residual,min(16,len(residual))).indices].cpu().tolist()))
        dimension=source_sig['embed_dim'];file=attempt/'features.npz'
        np.savez(file,features_1=np.asarray(outputs[0],dtype=np.float32).reshape(-1,dimension),
                 features_01=np.asarray(outputs[1],dtype=np.float32).reshape(-1,dimension),
                 ids=np.asarray([r['id'] for r in rows],dtype=str),classes=np.asarray([r['cls'] for r in rows],dtype=str))
        ab.write(attempt/'diagnostics.json',dict(image=stem,seconds=time.monotonic()-image_started,regions=diagnostics))
        ab.write(folder/'complete.json',dict(result=dict(file=str(file),source_signature=source_sig),
            artifacts=ab.identities([file,attempt/'diagnostics.json'])))
        print(f'[Extract {job["stage"]}] {stem}: {len(rows)} retained regions',flush=True)
    torch.cuda.synchronize()
    ab.write(Path(job['attempt'])/'worker_report.json',dict(status='complete',seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def worker(job):
    if job['stage'] in ('probe','extract'):return extract_worker(job)
    ab.PMFA=GMP;ab.EXPERIMENT_FORMAT=3004;ab.signature=signature
    rpg.signature=signature;rpg.pool=pool
    ab.install_pooling=lambda mode,params,stats:rpg.install_pooling(mode,params,stats,job)
    ab.worker(job)


def merge(cache,ids,raw,baseline_raw,rho):
    import numpy as np
    from src.feature_bank_storage import save_feature_matrix_npz
    wanted=read_bank(baseline_raw);features={};source=None
    for stem in ids:
        value=image_cache(Path(cache)/stem)
        if value is None:raise ValueError(f'Missing image cache: {stem}')
        if source is not None and source!=value['source_signature']:raise ValueError('Mixed image signatures')
        source=value['source_signature']
        with np.load(value['file'],allow_pickle=False) as data:
            for cls,sid,vec in zip(data['classes'],data['ids'],data['features_1' if rho==1. else 'features_01']):
                key=(str(cls),str(sid))
                if key in features:raise ValueError('Duplicate image cache member')
                features[key]=vec
    if set(features)!=set(wanted):raise ValueError('Fixed member mismatch')
    raw.mkdir(parents=True,exist_ok=False)
    for cls in sorted({k[0] for k in wanted}):
        keys=sorted(k for k in wanted if k[0]==cls)
        save_feature_matrix_npz(raw/'features'/cls/'features.npz',np.stack([features[k] for k in keys]),[k[1] for k in keys])
    ab.write(raw/'meta/feature_signature.json',signature(source,'gmp',GMP(rho)))
    shutil.copy2(Path(baseline_raw)/'meta/class_mapping.json',raw/'meta/class_mapping.json')
    ab.write(raw/'meta/fixed_members.json',dict(source=str(baseline_raw),count=len(features),filtering='baseline_members_no_refilter'))


def probe_summary(cache,ids):
    rows=[]
    for stem in ids:
        value=image_cache(Path(cache)/stem)
        if value is None:raise ValueError('Incomplete real probe')
        diagnostic=ab.read(Path(value['file']).parent/'diagnostics.json')
        rows.append(diagnostic)
    costs=[v['seconds'] for v in rows]
    solutions=[v for image in rows for v in image['regions'] if 'relative_residual' in v]
    if not solutions:raise ValueError('Probe had no nonzero retained regions')
    return dict(images=len(rows),seconds=sum(costs),mean_image_seconds=sum(costs)/len(costs),
                max_image_seconds=max(costs),max_relative_residual=max(v['relative_residual'] for v in solutions),
                solved_regions=len(solutions),note='Probe includes sampled SVD diagnostics; not pure GMP throughput.')


def choose(scores,baseline):
    if not all(math.isfinite(x) and 0<=x<=1 for x in [baseline,*scores.values()]):raise ValueError('Invalid internal score')
    best=max(scores,key=lambda k:(scores[k],k=='rho1'))
    return best if scores[best]>baseline else None


def report(base,current,ids):
    import numpy as np
    a,b=[ab.read(x['metrics']) for x in (base,current)]
    for value in (a,b):
        if value['evaluated_ids']!=ids or value['evaluated_count']!=len(ids) or value['eval_limit']!=0:
            raise ValueError('Incomplete full evaluation')
        matrix=np.asarray(value['confusion_matrix']);den=matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
        if matrix.shape!=(21,21) or (matrix<0).any() or not np.isfinite(matrix).all():raise ValueError('Invalid confusion')
        iou=matrix.diagonal()/den.clip(min=1)
        if not np.allclose(iou,list(value['per_class_iou'].values()),atol=2e-6,rtol=0) or abs(iou.mean()-value['mIoU'])>2e-6:
            raise ValueError('Metrics disagree with confusion')
    for key in ('cleanup','proposals_dir','proposal_count_before','proposal_count_after'):
        if a[key]!=b[key]:raise ValueError(f'Incompatible {key}')
    ignore={'INF_FEATURE_BANK_ROOT','INF_SAVE_DIR'}
    if {k:v for k,v in a['inference_settings'].items() if k not in ignore} != {k:v for k,v in b['inference_settings'].items() if k not in ignore}:
        raise ValueError('Inference settings differ')
    if not np.array_equal(np.asarray(a['confusion_matrix']).sum(1),np.asarray(b['confusion_matrix']).sum(1)):
        raise ValueError('Different evaluated ground truth')
    for result,metrics in ((base,a),(current,b)):
        if ab.read(Path(result['bank'])/'meta/feature_signature.json')!=metrics['feature_signature']:
            raise ValueError('Evaluated signature mismatch')
    if ab.bank_members(base['raw'])!=ab.bank_members(current['raw']):raise ValueError('Members changed')
    delta=100*(b['mIoU']-a['mIoU'])
    return dict(status='complete',smfa_miou=a['mIoU'],gmp_miou=b['mIoU'],delta_pp=delta,
        outcome='提高' if delta>1e-8 else '下降' if delta< -1e-8 else '持平',fixed_members=True,
        per_class_iou={c:dict(smfa=v,gmp=b['per_class_iou'][c],delta_pp=100*(b['per_class_iou'][c]-v)) for c,v in a['per_class_iou'].items()},
        evaluated_ids=ids,confusion_matrices=dict(smfa=a['confusion_matrix'],gmp=b['confusion_matrix']),
        note='One seed; candidate selected only on internal retrieval; verified historical SMFA baseline.')


def parallel(functions,cancel):
    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        try:
            pending={executor.submit(fn) for fn in functions}
            while pending:
                done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:future.result()
        except BaseException:
            cancel.set();raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--version',default='gmp_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    parser.add_argument('--max-probe-mean-seconds',type=float,default=120.,help='Stop if the 128-image probe averages above this budget')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    opts=parser.parse_args(argv);os.chdir(ROOT)
    if opts.worker:return worker(ab.read(opts.worker))
    if opts.previous is None:parser.error('--previous required')
    if not math.isfinite(opts.max_probe_mean_seconds) or opts.max_probe_mean_seconds<=0:parser.error('Probe budget must be finite and positive')
    if min(opts.gpus)<0 or len(set(opts.gpus))!=2:parser.error('Two distinct GPUs required')
    ab.parse_args(['--version',opts.version])
    opts.previous=opts.previous.resolve();rpg.verify_records=verify
    old,base,baseline_records=rpg.baseline(opts.previous)
    flags=['--version',opts.version,'--gpu',str(opts.gpus[0])]
    for key in ('dataset_root','image_dir','checkpoint','train_list','val_list','imagelevel_json','gt_dir'):
        flags+=['--'+key.replace('_','-'),old['options'][key]]
    for kind,flag in (('pseudo','--pseudo-dir'),('proposals','--proposals-dir')):
        flags += [flag,old['selected'].get(kind) or ab.read(opts.previous/kind/'complete.json')['result'][kind]]
    args=ab.parse_args(flags);context=ab.preflight(args);ab.check_gpu(opts.gpus[1])
    if not all(context['selected'].values()):raise ValueError('Complete shared inputs required; no regeneration')
    output=ROOT/'experiments'/opts.version;cache=ROOT/'storage/raw_banks/voc'/opts.version/'image_cache'
    logs=ROOT/'logs'/opts.version;weights_root=ROOT/'weights/sam3_adapter'/opts.version
    bank_root=ROOT/'storage/mlp_banks/voc'/opts.version
    identity=dict(inputs=context['inputs'],code=context['code'],baseline=baseline_records,gpus=opts.gpus,
                  previous=str(opts.previous),rho=[1.,.1],protocol='fixed_members_internal_selection_v1',
                  max_probe_mean_seconds=opts.max_probe_mean_seconds)
    if output.exists():
        if not opts.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version identity differs or --resume missing')
    elif any(p.exists() for p in (cache.parent,logs,weights_root,bank_root)):raise FileExistsError('Version already used')
    if opts.preflight_only:
        print('[Preflight passed] No SMFA/label/proposal generation; two fixed-member GMP candidates.',flush=True);return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    cancel=threading.Event();old_signal=signal.getsignal(signal.SIGTERM)
    def stop(*_):cancel.set();raise KeyboardInterrupt('GMP interrupted')
    signal.signal(signal.SIGTERM,stop);ab.SCRIPT=Path(__file__).resolve();stages=ab.Stages(output,opts.resume)
    from configs.experiment import CLEANUP
    common={k:str(getattr(args,k)) for k in ('dataset_root','image_dir','imagelevel_json','checkpoint','train_list','val_list','gt_dir')}
    common.update(images=context['images'],train_ids=context['train_ids'],val_ids=context['val_ids'],baseline=base,
                  pseudo=context['selected']['pseudo'],cache=str(cache),plan=str(output/'members.json'),mode='gmp')
    def execute(key,kind,gpu,**extras):
        if cancel.is_set():raise RuntimeError('Cancelled')
        def action(attempt):
            job=dict(common,attempt=str(attempt),stage=kind,**extras)
            env=ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT']=str(args.checkpoint)
            artifacts=[attempt];result={}
            if kind=='train':
                w=weights_root/key/attempt.name;b=bank_root/key/attempt.name
                job.update(weights=str(w),bank=str(b),split_file=str(Path(base['weights'])/'training_report.json'))
                artifacts += [w,b];result.update(weights=str(w),bank=str(b))
            if kind=='eval':
                job.update(metrics=str(attempt/'metrics.json'),cleanup=str(output/'cleanup.json'))
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(extras['weights'])/'adapter.pt'),
                    OVERRIDE_INF_FEATURE_BANK_ROOT=extras['bank'],OVERRIDE_INF_PROPOSALS_DIR=context['selected']['proposals'])
                result['metrics']=job['metrics']
            log=logs/f'{key}_{attempt.name}.log';ab.launch(job,attempt,env,log,cancel=cancel)
            if ab.read(attempt/'worker_report.json')['status']!='complete':raise RuntimeError('Worker failed')
            result['worker_report']=str(attempt/'worker_report.json')
            return result,artifacts+[log]
        return stages.run(key,action)
    try:
        ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,cache.parent,logs,weights_root,bank_root])),
            read_only_baseline=str(opts.previous),read_only_inputs=context['selected'],automatic_deletion=False))
        def plan_action(attempt):
            # Parent needs only mask extraction, not SAM3/CUDA.
            for key,value in ab.base_env(opts.gpus[0]).items():
                if key.startswith('OVERRIDE_'):os.environ[key]=value
            plan=enumerate_members(context['train_ids'],context['images'],context['selected']['pseudo'],base['raw'])
            ab.write(output/'members.json',plan);return {},[output/'members.json']
        stages.run('membership',plan_action)
        split=ab.read(Path(base['weights'])/'training_report.json')
        probe_ids=random.Random(42).sample(sorted(split['train_images']),min(128,len(split['train_images'])))
        execute('probe','probe',opts.gpus[0],ids=probe_ids,pmfa=asdict(GMP()))
        probe=probe_summary(cache,probe_ids);ab.write(output/'probe_summary.json',dict(ids=probe_ids,**probe))
        if probe['mean_image_seconds']>opts.max_probe_mean_seconds:
            raise RuntimeError('Real probe exceeded declared mean-image time budget; full extraction not started')
        parallel([lambda index=i:execute(f'extract_{index}','extract',opts.gpus[index],
            ids=context['train_ids'][index::2],pmfa=asdict(GMP())) for i in range(2)],cancel)
        candidates={}
        for name,rho in (('rho1',1.),('rho01',.1)):
            def merge_action(attempt,rho=rho,name=name):
                raw=cache.parent/name/attempt.name
                merge(cache,context['train_ids'],raw,base['raw'],rho)
                return dict(raw=str(raw)),[raw]
            candidates[name]=stages.run('merge_'+name,merge_action)
        def train(name,rho,gpu):
            candidates[name].update(execute('train_'+name,'train',gpu,raw=candidates[name]['raw'],pmfa=asdict(GMP(rho))))
        parallel([lambda:train('rho1',1.,opts.gpus[0]),lambda:train('rho01',.1,opts.gpus[1])],cancel)
        scores={}
        for name,value in candidates.items():
            training=ab.read(Path(value['weights'])/'training_report.json')
            if any(training[k]!=split[k] for k in ('train_images','validation_images')):raise ValueError('Internal split changed')
            scores[name]=training['best_macro_accuracy']
        winner=choose(scores,split['best_macro_accuracy'])
        ab.write(output/'selection.json',dict(scores=scores,smfa=split['best_macro_accuracy'],winner=winner,
                 criterion='internal_best_macro_accuracy',tie_break='rho1',final_val_used=False))
        verify(baseline_records,'Baseline');verify(context['inputs'],'Inputs');stages.verify_all()
        if ab.source_identity()!=context['code']:raise ValueError('Source changed')
        if winner is None:
            ab.write(output/'status.json',dict(status='complete_no_candidate_improvement',scores=scores))
            print('[DONE] Neither GMP candidate beat internal SMFA; no final val run.',flush=True);return
        ab.write(output/'cleanup.json',CLEANUP)
        current=candidates[winner];rho=1. if winner=='rho1' else .1
        current.update(execute('final_eval','eval',opts.gpus[0],bank=current['bank'],weights=current['weights'],pmfa=asdict(GMP(rho))))
        # A bad report is retryable without rerunning completed expensive stages.
        verify(baseline_records,'Baseline');verify(context['inputs'],'Inputs');stages.verify_all()
        if ab.source_identity()!=context['code']:raise ValueError('Source changed')
        value=report(base,current,context['val_ids']);value.update(selected=winner,rho=rho,internal_scores=scores,
            baseline_source=str(opts.previous),stages={p.parent.name:ab.read(p)['seconds'] for p in output.glob('*/complete.json')})
        value['workers']={p.parent.name:ab.read(ab.read(p)['result']['worker_report']) for p in output.glob('*/complete.json') if 'worker_report' in ab.read(p)['result']}
        ab.write(output/'comparison.json',value)
        (output/'per_class.csv').write_text('class,smfa_iou,gmp_iou,delta_pp\n'+''.join(
            f'{c},{v["smfa"]},{v["gmp"]},{v["delta_pp"]}\n' for c,v in value['per_class_iou'].items()),encoding='utf-8')
        (output/'comparison.md').write_text(f'SMFA: {100*value["smfa_miou"]:.5f}%\n\nGMP: {100*value["gmp_miou"]:.5f}%\n\nDelta: {value["delta_pp"]:+.5f} pp\n',encoding='utf-8')
        ab.write(output/'status.json',dict(status='complete',winner=winner))
        print(f'[DONE] {output}/comparison.json',flush=True)
    except BaseException as exc:
        cancel.set();ab.write(output/'status.json',dict(status='incomplete',error=str(exc)));raise
    finally:
        signal.signal(signal.SIGTERM,old_signal);lock.unlink()


if __name__=='__main__':main()
