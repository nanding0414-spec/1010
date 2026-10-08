#!/usr/bin/env python3
"""Read-only VOC error diagnosis. GT-assisted outputs are NOT deployable scores."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_mask_score_queue as old
from tools import experimental_error_oracles as base
from project_utils.experiment_notify import notify_experiment
ab, queue, gmp = old.ab, old.queue, old.gmp
PROTOCOL = 'cleanup_retrieval_audit_v1'

from tools.experimental_error_attribution import checked_history

def recipes(config):
    """Leave one operation/constraint out; no validation-driven search."""
    result={'baseline':dict(config)}
    for name,key,value in [('no_score','min_score',0.),('no_area','min_mask_area',0),
        ('no_islands','min_island_area',0),('no_hole_fill','max_hole_area',0),
        ('no_hole_fraction_cap','max_hole_fraction',1.),('no_cleanup_nms','nms_iou',1.)]:
        result[name]={**config,key:value}
    return result


def coverage_counts(gt,masks,reference):
    union=np.zeros(gt.shape,bool)
    for mask in masks:
        if mask.shape!=gt.shape:raise ValueError('Mask/GT size mismatch')
        union |= mask.astype(bool)
    if not np.isin(gt,list(range(21))+[255]).all():raise ValueError('Invalid VOC labels')
    return {str(c):dict(pixels=int((gt==c).sum()),covered=int(((gt==c)&union).sum()),
        gained=int(((gt==c)&union&~reference).sum()),lost=int(((gt==c)&~union&reference).sum())) for c in range(21)}


def cleanup_audit(common,raw_dir,attempt,cancel,deadline):
    from src.proposal_cleanup import CleanupConfig,clean_proposals
    from PIL import Image
    meta=ab.read(Path(common['proposals'])/'cleanup_config.json');configs=recipes(meta['config'])
    totals={};per_image=attempt/'coverage.jsonl'
    with per_image.open('x',encoding='utf-8') as stream:
        for pos,stem in enumerate(common['val_ids'],1):
            if cancel.is_set():raise KeyboardInterrupt('Cancelled')
            queue.ensure_resources(deadline)
            with np.load(raw_dir/(stem+'.npz'),allow_pickle=False) as data:
                masks=data['masks'];scores=data['scores']
            with np.load(Path(common['proposals'])/(stem+'.npz'),allow_pickle=False) as data:
                existing=data['masks'].astype(bool);es=data['scores'];ei=data['source_indices']
            with Image.open(Path(common['gt_dir'])/(stem+'.png')) as im:gt=np.array(im)
            baseline,bs,bi,_=clean_proposals(masks,scores,CleanupConfig(**configs['baseline']))
            if not all(np.array_equal(x,y) for x,y in [(baseline,existing),(bs,es),(bi,ei)]):
                raise ValueError('Cleaning baseline failed exact cache replay: '+stem)
            reference=np.zeros(gt.shape,bool)
            for m in baseline:reference |= m
            image_result={}
            for name,config in configs.items():
                cleaned,_,_,changes=clean_proposals(masks,scores,CleanupConfig(**config))
                counts=coverage_counts(gt,cleaned,reference)
                image_result[name]=dict(count=len(cleaned),coverage=counts,changes=changes)
                if name not in totals:totals[name]=dict(count=0,coverage={c:{k:0 for k in row} for c,row in counts.items()})
                totals[name]['count']+=len(cleaned)
                for c,row in counts.items():
                    for k,v in row.items():totals[name]['coverage'][c][k]+=v
            import json
            stream.write(json.dumps(dict(image=stem,variants=image_result))+'\n')
            if pos%50==0:print(f'[Cleanup audit] {pos}/1449',flush=True)
    path=attempt/'cleanup_summary.json'
    ab.write(path,dict(configurations=configs,results=totals,evaluation_ids=common['val_ids'],
        note='Union coverage only. Background covered is not final model FP. Counterfactual rule effects are not additive.'))
    return dict(report=str(path)),[attempt]


def decompose(query,context,labels,sims,indices,topm,weight):
    """Mirror balanced retrieval incl. centroid fallback for absent local classes."""
    query=np.asarray(query,np.float32)
    centers=query@context.centroids.T
    local=np.empty_like(centers);support=np.zeros_like(centers,dtype=np.int32)
    for row,(ss,ii) in enumerate(zip(sims,indices)):
        valid=ii>=0; ss=ss[valid];ii=ii[valid]
        for c,name in enumerate(context.labels):
            values=ss[labels[ii]==name];support[row,c]=len(values)
            k=min(max(int(topm),1),len(values))
            local[row,c]=float(np.partition(values,-k)[-k:].mean()) if k else float(centers[row,c])
    # Python scalar arithmetic then float32 matches the production per-class assignment.
    fused=np.asarray([[(1.-weight)*float(l)+weight*float(c) for l,c in zip(ll,cc)]
        for ll,cc in zip(local,centers)],np.float32)
    return local,centers,fused,support


def install(job):
    original=queue.install_cached_eval
    def hook(j):
        original(j)
        import run_inference as inference
        import src.inference as inf
        import src.retrieval as retrieval
        from PIL import Image
        load=inference.load_strict_proposals;build=inference.build_feature_extractor
        classify=inf.classify_queries;search=retrieval._search;raster=inference.rasterize_instances
        state={};cms=[];details={};score_rows=[]
        def load_hook(directory,stem,height,width):
            masks,scores=load(directory,stem,height,width)
            state.clear();state.update(stem=stem,masks=masks,valid_ids=[],scores=[])
            return masks,scores
        def build_hook(*args,**kwargs):
            model=build(*args,**kwargs);extract=model.extract_features_batch
            def record(rgb,masks):
                vectors=extract(rgb,masks)
                state['valid_ids']=[i for i,v in enumerate(vectors) if np.isfinite(v).all() and np.linalg.norm(v)>=1e-6]
                return vectors
            model.extract_features_batch=record
            return model
        def search_hook(*args,**kwargs):
            value=search(*args,**kwargs);state['search']=value;return value
        def classify_hook(*args,**kwargs):
            result=classify(*args,**kwargs)
            if kwargs.get('mode')!='balanced':raise ValueError('Balanced retrieval required')
            labels=np.asarray([retrieval.canonical_label(v) for v in kwargs['all_labels']])
            ss,ii=state.pop('search');ctx=kwargs['context']
            local,center,fused,support=decompose(kwargs['query_feats'],ctx,labels,ss,ii,kwargs['class_topm'],kwargs['centroid_weight'])
            if len(result)!=len(state['valid_ids']):raise ValueError('Candidate row alignment mismatch')
            for k,(r,index) in enumerate(zip(result,state['valid_ids'])):
                order=np.argsort(fused[k])[::-1];expected=ctx.labels[order[0]]
                if r.label!=expected or abs(r.score-float(fused[k,order[0]]))>2e-6 or abs(r.margin-float(fused[k,order[0]]-fused[k,order[1]]))>2e-6:
                    raise ValueError('Score decomposition differs from original retrieval')
                lp=ctx.labels[np.argsort(local[k])[::-1][0]];cp=ctx.labels[np.argsort(center[k])[::-1][0]]
                state['scores'].append(dict(index=index,classes=list(ctx.labels),local=local[k].tolist(),
                    center=center[k].tolist(),fused=fused[k].tolist(),support=support[k].tolist(),
                    local_top1=lp,center_top1=cp,final_top1=r.label,margin=r.margin,
                    note='Local-only uses original centroid fallback for classes absent from search candidates'))
            return result
        def raster_hook(rows,height,width):
            queue.ensure_resources(float('inf'));prediction=raster(rows,height,width);stem=state['stem']
            if stem in details:raise ValueError('Duplicate image')
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im:gt=np.array(im)
            for row in state['scores']:
                mask=state['masks'][row['index']];hist=np.bincount(gt[mask&(gt!=255)],minlength=21)
                majority=int(hist.argmax()) if hist.sum() else None
                row['pseudo_reference']=False
                row['gt_majority']=('background' if majority==0 else inf.DATASET_CLASSES[majority-1]) if majority is not None else None
                row['gt_histogram']=hist.tolist();row['image']=stem
                score_rows.append(row)
            cms.append(base.confusion(gt,prediction));details[stem]={}
            return prediction
        inference.load_strict_proposals=load_hook;inference.build_feature_extractor=build_hook
        inference.rasterize_instances=raster_hook;inf.classify_queries=classify_hook;retrieval._search=search_hook
        job['_cms']=cms;job['_details']=details;job['_scores']=score_rows;job['_classes']=list(inf.DATASET_CLASSES)
    queue.install_cached_eval=hook


def worker(job):
    install(job);old.previous.worker(job)
    ids=list(job['_details'])
    if ids!=job['val_ids'] or len(ids)!=1449:raise ValueError('Incomplete evaluation')
    rows=job['_scores'];valid=[r for r in rows if r['gt_majority'] is not None]
    flips=[r for r in valid if r['local_top1']!=r['final_top1']]
    chair=[r for r in valid if r['gt_majority']=='chair' and r['final_top1']=='sofa']
    summary=dict(candidates=len(rows),local_final_disagreements=len(flips),
        local_correct_final_wrong=sum(r['local_top1']==r['gt_majority'] and r['final_top1']!=r['gt_majority'] for r in flips),
        local_wrong_final_correct=sum(r['local_top1']!=r['gt_majority'] and r['final_top1']==r['gt_majority'] for r in flips),
        chair_to_sofa=chair,note='GT-majority candidate labels are diagnostic only; candidate accuracy is not mIoU.')
    dest=Path(job['attempt']);cm=np.stack(job['_cms'])
    np.savez_compressed(dest/'oracle_per_image.npz',ids=np.asarray(ids),baseline=cm)
    ab.write(dest/'all_candidate_scores.json',dict(rows=rows))
    ab.write(dest/'diagnostics.json',dict(classes=['background']+job['_classes'],score_summary=summary,
        results=dict(baseline=base.metrics(cm.sum(0)))))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    p.add_argument('--clean-proposals-dir',type=Path,default=Path(old.mpsa.DEFAULT_CLEAN))
    p.add_argument('--version',default='mechanism_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1]);p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or not np.isfinite(args.max_hours) or args.max_hours<=0: p.error('Invalid devices/budget')
    for gpu in args.gpus: ab.check_gpu(gpu)
    common,refs,split,records,clean,raw,sig=checked_history(args.previous.resolve(),args.clean_proposals_dir.resolve())
    if len(common['val_ids'])!=1449: raise ValueError('Expected full VOC val')
    raw_proposals=Path(ab.read(args.clean_proposals_dir/'cleanup_config.json')['input_dir'])
    records+=ab.identities([raw_proposals/(i+'.npz') for i in common['val_ids']])
    code=ab.source_identity()
    identity=old.canonical(dict(protocol=PROTOCOL,previous=str(args.previous.resolve()),references=records,code=code,gpus=args.gpus,
        clean=str(args.clean_proposals_dir.resolve()),seeds=list(old.SEEDS),boundary_radius=3,component_connectivity=4,audit='fixed_leave_one_rule_out_and_score_decomposition_v1'))
    output=ROOT/'experiments'/args.version;logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version differs or --resume missing')
    elif logs.exists():raise FileExistsError(logs)
    stages=ab.Stages(output,True);stages.verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+2**30:raise queue.Deferred('Need 1 GiB allowance + 5 GiB reserve')
    print('[Plan] CPU cleanup rule audit (in memory), then three cached historical model replays and full score export. No training or SAM3.',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists():ab.write(budget,dict(started_at=time.time()))
    deadline=ab.read(budget)['started_at']+args.max_hours*3600
    cancel=threading.Event();handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Cancelled')
    for s in handlers:signal.signal(s,stop)
    entry=queue.__file__;queue.__file__=__file__;status='running';error=None;reports={};guard=threading.Lock()
    def execute(seed,gpu):
        def action(attempt):
            model=refs[seed]
            job=dict(common,raw_signature=sig,group='C',role='gmp2',variant='d2',stage='eval',seed=seed,
                     raw_proposals=str(raw_proposals),weights=model['weights'],bank=model['bank'],cleanup='PRE_CLEANED',attempt=str(attempt),metrics=str(attempt/'metrics.json'))
            env=ab.base_env(gpu)
            env.update(OVERRIDE_SAM3_CKPT=common['checkpoint'],OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT='0.2',
                OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(model['weights'])/'adapter.pt'),
                OVERRIDE_INF_FEATURE_BANK_ROOT=model['bank'],OVERRIDE_INF_PROPOSALS_DIR=common['proposals'])
            log=logs/f'seed{seed}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            metric=ab.read(job['metrics'])
            old.validate_eval(metric,ab.read(model['metrics']),common['val_ids'],clean['count'],.2)
            old.mpsa.require_baseline_replay(ab.read(model['metrics']),metric)
            old.require_image_replay(model['per_image'],str(attempt/'per_image_confusion.npz'))
            diag=ab.read(attempt/'diagnostics.json')
            if diag['results']['baseline']['confusion_matrix']!=metric['confusion_matrix']:raise ValueError('Diagnostic baseline mismatch')
            return dict(diagnostics=str(attempt/'diagnostics.json'),per_image=str(attempt/'oracle_per_image.npz')), [attempt,log]
        value=stages.run(f'seed_{seed}',action)
        with guard:
            reports[str(seed)]=value
            ab.write(output/'status.json',dict(status='running',completed=sorted(reports)))
    try:
        ab.write(output/'status.json',dict(status=status))
        ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(logs)],automatic_deletion=False))
        audit_result=stages.run('cleanup_audit',lambda attempt:cleanup_audit(common,raw_proposals,attempt,cancel,deadline))
        queue.run_round(list(old.SEEDS),args.gpus,execute,cancel,deadline,lambda item,exc:(_ for _ in ()).throw(exc))
        gmp.verify(records,'Historical inputs')
        if ab.source_identity()!=code:raise ValueError('Code changed during experiment')
        results={s:ab.read(r['diagnostics']) for s,r in reports.items()}
        summary={s:r['score_summary'] for s,r in results.items()}
        ab.write(output/'summary.json',dict(status='complete',protocol=PROTOCOL,score_decomposition=summary,
            cleanup_audit=ab.read(audit_result['report']),reports=reports,evaluation_ids=common['val_ids'],
            note='Coverage is not segmentation accuracy. Score alternatives are diagnostic ranks, not deployed predictions.'))
        (output/'summary.md').write_text('Completed cleanup and retrieval score audit. See summary.json for coverage and class flips.\n',encoding='utf-8')
        status='complete'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except KeyboardInterrupt as exc:status='interrupted';error=str(exc);cancel.set();raise
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        ab.write(output/'status.json',dict(status=status,error=error,completed=sorted(reports)))
        queue.__file__=entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink()
        notify_experiment(args.version,status,str(output),time.time()-ab.read(budget)['started_at'])
    print(f'[{status}] {output}',flush=True)


if __name__=='__main__':main()
