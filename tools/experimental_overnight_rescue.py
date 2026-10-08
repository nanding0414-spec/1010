#!/usr/bin/env python3
"""12-hour fixed rescue/retrieval batch. No retraining or validation-driven search."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_mask_score_queue as old
from tools import experimental_error_oracles as oracle
from tools.experimental_error_attribution import checked_history
from project_utils.experiment_notify import notify_experiment
ab,queue,gmp=old.ab,old.queue,old.gmp
PROTOCOL='overnight_semantic_rescue_v1'
METHODS=('baseline','local_only','loose','rescue','rescue_strict')
RECIPE=dict(methods=METHODS,seeds=[42,123,2026],relaxed_min_score=0.,
    rescue=dict(proposal_score=.4,retrieval_score=.85,margin=.05,novel_fraction=.1),
    rescue_strict=dict(proposal_score=.4,retrieval_score=.9,margin=.1,novel_fraction=.1),
    insertion='background_only_after_original_foreground_NMS',features='single_phase_hard_center_GMP_rho01',
    local_only='centroid_weight_0_retaining_absent_class_centroid_fallback')


def pool(job,stem):
    from src.proposal_cleanup import CleanupConfig,clean_proposals
    cp=Path(job['proposals'])/(stem+'.npz');rp=Path(job['raw_proposals'])/(stem+'.npz')
    if ab.digest(cp)!=job['cache_identity'][stem]['mask_source_sha256'] or ab.digest(rp)!=job['raw_hashes'][stem]:
        raise ValueError('Proposal identity changed')
    with np.load(cp,allow_pickle=False) as data:strong=list(data['masks'].astype(bool));ss=data['scores'].copy()
    with np.load(rp,allow_pickle=False) as data:
        cfg=dict(job['cleanup_recipe'],min_score=0.)
        relaxed,rs,indices,_=clean_proposals(data['masks'],data['scores'],CleanupConfig(**cfg))
    known=set(queue.mask_hashes(strong));extra=[];scores=[];sources=[]
    for mask,score,index in zip(relaxed,rs,indices):
        digest=queue.mask_hashes([mask])[0]
        if digest in known:continue
        known.add(digest);extra.append(mask);scores.append(float(score));sources.append(int(index))
    return strong,ss,extra,np.asarray(scores,np.float32),sources


def rescue_rows(strong,extras,params):
    """Strong predictions are immutable. Rescues fill only uncovered background."""
    occupied=np.zeros((strong or extras)[0]['mask'].shape,bool) if strong or extras else None
    for row in strong:
        if row['cls_name']!='background':occupied |= row['mask']
    result=list(strong);accepted=0
    for row in sorted(extras,key=lambda r:-r['score']):
        if row['cls_name']=='background' or row['proposal_score']<params['proposal_score'] or row['retrieval_score']<params['retrieval_score'] or row['retrieval_margin']<params['margin']:continue
        mask=row['mask'] & ~occupied
        if not mask.any() or mask.sum()/max(row['mask'].sum(),1)<params['novel_fraction']:continue
        result.append(dict(row,mask=mask));occupied |= mask;accepted+=1
    return result,accepted


def cached_extra(job,stem):
    marker=Path(job['new_cache'])/stem/'complete.json';value=ab.read(marker)
    gmp.verify(value['artifacts'],'Extra candidate features')
    with np.load(value['result']['file'],allow_pickle=False) as data:
        return data['vectors'].copy(),data['hashes'].tolist()


def extract(job):
    import torch
    from PIL import Image
    sys.path.insert(0,str(ROOT/'sam3-main'));ab.install_precision()
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    if ab.digest(job['checkpoint'])!=job['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    model=None;stages=ab.Stages(job['new_cache'],True)
    for stem in job['ids']:
        queue.ensure_resources(job['deadline'])
        def action(attempt):
            nonlocal model
            strong,ss,extra,es,sources=pool(job,stem)
            image_path=job['images'][stem]
            if ab.digest(image_path)!=job['cache_identity'][stem]['image_sha256']:raise ValueError('Image changed')
            vectors=[]
            if extra or job['stage']=='probe':
                if model is None:
                    model=SAM3FeatureExtractor(model_path=job['checkpoint'],device=torch.device('cuda:0'),resolution=1008,
                        feature_level=-1,feature_source='trunk_fpn',token_norm='layernorm',context_subtract=0,
                        use_bf16=False,use_interactivity=False,enable_segmentation=False)
                with Image.open(image_path) as im:rgb=np.array(im.convert('RGB'))
                inp,_=model._preprocess_image(rgb)
                with torch.inference_mode():tokens,grid=model._forward_tokens(inp)
                if tuple(grid)!=(72,72) or tuple(tokens.shape)!=(1,5184,1280) or not torch.isfinite(tokens).all():raise ValueError('Invalid feature geometry/numerics')
                def vector(mask):
                    index,_=old.mpsa.selection(mask,(0,0));t=tokens[0,torch.as_tensor(index,device=tokens.device,dtype=torch.long)].float()
                    z,_=gmp.solve(t,torch.ones(len(index),device=tokens.device),.1)
                    return z.cpu().numpy()
                reference=queue.cache_vectors(Path(job['cache'])/'val'/stem,job['cache_identity'][stem],'C',queue.mask_hashes(strong))
                for i,m in enumerate(strong[:2]):
                    if not np.allclose(vector(m),reference[i],atol=5e-5,rtol=5e-5):raise ValueError('FP32 GMP does not reproduce history')
                vectors=[vector(m) for m in extra]
            array=np.asarray(vectors,np.float32).reshape(-1,1280)
            if not np.isfinite(array).all() or (np.linalg.norm(array,axis=1)<1e-6).any():raise ValueError('Invalid extra features')
            file=attempt/'extras.npz';np.savez_compressed(file,vectors=array,hashes=np.asarray(queue.mask_hashes(extra),dtype=str),source_indices=sources)
            return dict(file=str(file),extra_count=len(extra)),[file]
        stages.run(stem,action)
        print('[Extra features]',stem,flush=True)


def install(job):
    original=queue.install_cached_eval
    def hook(j):
        # Suppress old five-image logger: its row mapping only supports old masks.
        queue.KEY_IMAGES=[];original(j)
        import run_inference as inference
        import src.inference as inf
        from PIL import Image
        load=inference.load_strict_proposals;read_vectors=queue.cache_vectors
        nms=inf.nms_masks_per_class;raster=inference.rasterize_instances
        state={};ids=[];cms=[];counts=[]
        reference_matrices={}
        if job['method'] in ('rescue','rescue_strict'):
            with np.load(job['baseline_per_image'],allow_pickle=False) as data:
                reference_matrices={str(i):cm.copy() for i,cm in zip(data['ids'],data['confusions'])}
        expanded=job['method'] in ('loose','rescue','rescue_strict')
        def load_hook(directory,stem,height,width):
            masks,scores=load(directory,stem,height,width)
            state.clear();state.update(stem=stem,strong_hashes=set(queue.mask_hashes(masks)),accepted=0)
            if not expanded:return masks,scores
            strong,ss,extra,es,_=pool(job,stem)
            if queue.mask_hashes(strong)!=queue.mask_hashes(masks):raise ValueError('Strong cache changed')
            state.update(extra=extra,strong=strong)
            return strong+extra,list(ss)+list(es)
        def vectors_hook(folder,identity,group,hashes):
            if not expanded:return read_vectors(folder,identity,group,hashes)
            if group!='C' or Path(folder).name!=state['stem']:raise ValueError('Unexpected feature request')
            strong=state['strong'];extra=state['extra'];eh=queue.mask_hashes(extra)
            if hashes!=queue.mask_hashes(strong)+eh:raise ValueError('Candidate order mismatch')
            added,stored=cached_extra(job,state['stem'])
            if stored!=eh:raise ValueError('Extra masks differ from extraction')
            base=read_vectors(folder,identity,group,queue.mask_hashes(strong))
            return np.concatenate([base,added],axis=0)
        def nms_hook(rows,threshold):
            if job['method'] not in ('rescue','rescue_strict'):return nms(rows,threshold)
            strong=[];extra=[]
            for row in rows:
                (strong if queue.mask_hashes([row['mask']])[0] in state['strong_hashes'] else extra).append(row)
            kept=nms(strong,threshold)
            # Ground truth is used later only to verify the unchanged strong path.
            ordered=sorted(kept,key=lambda r:-r['score'])
            shape=(state['strong']+state['extra'])[0].shape if state['strong'] or state['extra'] else None
            if shape is not None:
                mapping={c:i+1 for i,c in enumerate(inf.DATASET_CLASSES)}
                state['strong_prediction']=oracle.compose([r['mask'] for r in ordered],
                    [mapping.get(r['cls_name'],0) for r in ordered],shape)
            result,accepted=rescue_rows(kept,extra,RECIPE[job['method']]);state['accepted']=accepted
            return result
        def raster_hook(rows,height,width):
            queue.ensure_resources(float('inf'));pred=raster(rows,height,width);stem=state['stem']
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im:gt=np.array(im)
            if reference_matrices:
                strong_pred=state.get('strong_prediction',np.zeros_like(pred))
                if not np.array_equal(oracle.confusion(gt,strong_pred),reference_matrices[stem]):raise ValueError('Rescue strong path differs from history')
                if not np.array_equal(pred[strong_pred>0],strong_pred[strong_pred>0]):raise ValueError('Rescue overwrote original foreground')
            ids.append(stem);cms.append(oracle.confusion(gt,pred));valid=gt!=255
            counts.append(dict(image=stem,background_fp=int(((gt==0)&(pred>0)).sum()),
                foreground_fn=int(((gt>0)&valid&(pred==0)).sum()),accepted_rescues=state['accepted']))
            return pred
        inference.load_strict_proposals=load_hook;queue.cache_vectors=vectors_hook
        inf.nms_masks_per_class=nms_hook;inference.rasterize_instances=raster_hook
        job['_report']=(ids,cms,counts,['background']+list(inf.DATASET_CLASSES))
    queue.install_cached_eval=hook


def worker(job):
    import torch
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();started=time.monotonic();status='failed';error=None;category='global'
    try:
        if job['stage'] in ('extract','probe'):extract(job)
        else:
            install(job);old.previous.worker(job)
            ids,cms,counts,classes=job.pop('_report')
            if ids!=job['val_ids']:raise ValueError('Incomplete evaluation')
            ab.write(Path(job['attempt'])/'result.json',dict(method=job['method'],seed=job['seed'],classes=classes,
                metrics=oracle.metrics(np.sum(cms,axis=0)),images=counts,
                effective_candidate_pool='original_clean' if job['method'] in ('baseline','local_only') else 'original_clean_union_relaxed_score0_deduplicated_by_mask',
                recipe=RECIPE))
        torch.cuda.synchronize();status='complete'
    except torch.cuda.OutOfMemoryError as exc:category='oom';error=str(exc);raise
    except (ValueError,FileNotFoundError,KeyError) as exc:error=str(exc);raise
    except queue.Deferred as exc:category='deferred';error=str(exc);raise
    except BaseException as exc:category='local' if job['stage']=='eval' else 'global';error=str(exc);raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,error=error,category=category,
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    p.add_argument('--clean-proposals-dir',type=Path,default=Path(old.mpsa.DEFAULT_CLEAN))
    p.add_argument('--version',default='overnight_rescue_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1]);p.add_argument('--max-hours',type=float,default=12)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or not math.isfinite(args.max_hours) or args.max_hours<=0:p.error('Invalid devices/budget')
    for gpu in args.gpus:ab.check_gpu(gpu)
    common,refs,split,records,clean,raw,sig=checked_history(args.previous.resolve(),args.clean_proposals_dir.resolve())
    meta=ab.read(args.clean_proposals_dir/'cleanup_config.json');rawdir=Path(meta['input_dir'])
    rawhash={s:ab.digest(rawdir/(s+'.npz')) for s in common['val_ids']}
    records+=ab.identities([rawdir/(s+'.npz') for s in common['val_ids']])
    code=ab.source_identity();identity=old.canonical(dict(protocol=PROTOCOL,recipe=RECIPE,code=code,references=records,
        previous=str(args.previous.resolve()),gpus=args.gpus,cleanup_recipe=meta['config']))
    output=ROOT/'experiments'/args.version;cache=ROOT/'storage/raw_banks/voc'/args.version/'rescue';logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version mismatch or --resume missing')
    elif cache.parent.exists() or logs.exists():raise FileExistsError('Version already used')
    stages=ab.Stages(output,True);stages.verify_all();ab.Stages(cache,True).verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+2*2**30:raise queue.Deferred('Need 2 GiB allowance + 5 GiB reserve')
    print('[Plan] 15 full evaluations: baseline/local-only/loose/rescue/rescue-strict x 3 seeds; one shared extra-feature extraction.',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists():ab.write(budget,dict(started_at=time.time()))
    deadline=ab.read(budget)['started_at']+args.max_hours*3600;cancel=threading.Event();reports={};failures={};guard=threading.Lock()
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Cancelled')
    for s in handlers:signal.signal(s,stop)
    entry=queue.__file__;queue.__file__=__file__;status='running';error=None;expanded_count=None
    def flush():
        values={k:ab.read(v['report']) for k,v in reports.items()};summary={}
        for method in METHODS:
            rows={str(s):values[f'{method}_{s}'] for s in old.SEEDS if f'{method}_{s}' in values}
            scores=[r['metrics']['mIoU'] for r in rows.values()]
            summary[method]=dict(complete=len(rows)==3,seeds={s:dict(metrics=r['metrics'],
                background_fp=sum(x['background_fp'] for x in r['images']),foreground_fn=sum(x['foreground_fn'] for x in r['images']),
                accepted_rescues=sum(x['accepted_rescues'] for x in r['images']),
                delta_pp=100*(r['metrics']['mIoU']-values[f'baseline_{s}']['metrics']['mIoU']) if f'baseline_{s}' in values else None) for s,r in rows.items()},
                mean=float(np.mean(scores)) if len(rows)==3 else None,sample_std=float(np.std(scores,ddof=1)) if len(rows)==3 else None)
        ab.write(output/'summary.json',dict(status=status,protocol=PROTOCOL,recipe=RECIPE,results=summary,failures=failures,
            reports=reports,evaluation_ids=common['val_ids'],classes=next(iter(values.values()))['classes'] if values else None,
            note='Fixed VOC val development batch. Incomplete groups have no reported mean; no automatic deployment.'))
        ab.write(output/'status.json',dict(status=status,error=error,completed=sorted(reports),failures=failures))
        lines=[f'Status: {status}', '', '| Method | Completed seeds | Mean mIoU % |','|---|---:|---:|']
        for method,r in summary.items():
            mean='incomplete' if r['mean'] is None else f'{100*r["mean"]:.4f}'
            lines.append(f'| {method} | {len(r["seeds"])}/3 | {mean} |')
        (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def execute(key,stage,gpu,ids=None,method='baseline',seed=42):
        def action(attempt):
            model=refs[seed];job=dict(common,stage=stage,attempt=str(attempt),ids=ids,method=method,seed=seed,
                role='gmp2',group='C',variant='d2',raw_signature=sig,new_cache=str(cache),raw_proposals=str(rawdir),
                raw_hashes=rawhash,cleanup_recipe=meta['config'],deadline=deadline,weights=model['weights'],bank=model['bank'],
                baseline_per_image=model['per_image'],
                cleanup='PRE_CLEANED',metrics=str(attempt/'metrics.json'))
            env=ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT']=common['checkpoint']
            if stage=='eval':env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(model['weights'])/'adapter.pt'),
                OVERRIDE_INF_FEATURE_BANK_ROOT=model['bank'],OVERRIDE_INF_PROPOSALS_DIR=common['proposals'],
                OVERRIDE_INF_CENTROID_WEIGHT='0' if method=='local_only' else '0.4')
            log=logs/(key+'_'+attempt.name+'.log');queue.launch(job,attempt,env,log,cancel,deadline)
            artifacts=[attempt,log];result={}
            if stage=='eval':
                metric=ab.read(job['metrics']);ref=old.canonical(ab.read(model['metrics']))
                count=clean['count'] if method in ('baseline','local_only') else expanded_count
                # Explicitly declared pool/scoring differences; all other settings still strict.
                ref['proposal_count_before']=ref['proposal_count_after']=count
                if method=='local_only':ref['inference_settings']['INF_CENTROID_WEIGHT']=0.
                old.validate_eval(metric,ref,common['val_ids'],count,.2)
                old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
                if metric['feature_signature']!=ab.read(Path(model['bank'])/'meta/feature_signature.json'):raise ValueError('Adapter/bank signature differs')
                detail=ab.read(attempt/'result.json')
                if detail['metrics']['confusion_matrix']!=metric['confusion_matrix']:raise ValueError('Metric mismatch')
                if method=='baseline':
                    old.mpsa.require_baseline_replay(ab.read(model['metrics']),metric)
                    old.require_image_replay(model['per_image'],str(attempt/'per_image_confusion.npz'))
                result=dict(report=str(attempt/'result.json'),per_image=str(attempt/'per_image_confusion.npz'))
            else:artifacts += [cache/stem for stem in ids]
            return result,artifacts
        return stages.run(key,action)
    def evaluate(item,gpu):
        method,seed=item;key=f'{method}_{seed}';value=execute(key,'eval',gpu,method=method,seed=seed)
        with guard:reports[key]=value;flush()
    def fatal(item,exc):raise exc
    def local_failure(item,exc):
        if isinstance(exc,queue.GlobalFailure):raise exc
        with guard:failures[str(item)]=str(exc);flush()
    try:
        flush();ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,cache.parent,logs])),automatic_deletion=False))
        queue.run_round([('baseline',s) for s in old.SEEDS],args.gpus,evaluate,cancel,deadline,fatal)
        queue.run_round([('local_only',s) for s in old.SEEDS],args.gpus,evaluate,cancel,deadline,local_failure)
        execute('real_probe','probe',args.gpus[0],common['val_ids'][:1])
        queue.run_round([0,1],args.gpus,lambda shard,gpu:execute(f'extract_{shard}','extract',gpu,common['val_ids'][shard::2]),cancel,deadline,fatal)
        expanded_count=clean['count']+sum(ab.read(cache/s/'complete.json')['result']['extra_count'] for s in common['val_ids'])
        queue.run_round([(m,s) for s in old.SEEDS for m in ('loose','rescue','rescue_strict')],args.gpus,evaluate,cancel,deadline,local_failure)
        gmp.verify(records,'Historical inputs')
        if ab.source_identity()!=code:raise ValueError('Source changed')
        status='complete' if len(reports)==15 else 'partial'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except KeyboardInterrupt as exc:status='interrupted';error=str(exc);cancel.set();raise
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();queue.__file__=entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink();notify_experiment(args.version,status,str(output),time.time()-ab.read(budget)['started_at'])


if __name__=='__main__':main()
