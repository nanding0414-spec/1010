#!/usr/bin/env python3
"""Fixed rescue mechanism batch: 16 methods x 3 seeds; verified read-only feature reuse."""
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
PROTOCOL='rescue_mechanisms_v1'
from tools import experimental_overnight_rescue as prior
pool, cached_extra = prior.pool, prior.cached_extra


def parameters(**overrides):
    return dict(dict(proposal_score=.4, retrieval_score=.9, margin=.1,
                     novel_fraction=.1, conflict=1., votes=1, order='score'), **overrides)


VARIANTS={
    'rescue_strict': parameters(),
    'background_only': parameters(proposal_score=0.,retrieval_score=-2.,margin=0.,novel_fraction=0.),
    'score_only': parameters(retrieval_score=-2.,margin=0.,novel_fraction=0.),
    'score_retrieval': parameters(margin=0.,novel_fraction=0.),
    'score_margin': parameters(retrieval_score=-2.,novel_fraction=0.),
    'no_sam_score': parameters(proposal_score=0.),
    'no_novelty': parameters(novel_fraction=0.),
    'novelty30': parameters(novel_fraction=.3),
    'score60': parameters(proposal_score=.6),
    'margin20': parameters(margin=.2),
    'conflict10': parameters(conflict=.1),
    'conflict30': parameters(conflict=.3),
    'vote2': parameters(votes=2),
    'conflict10_vote2': parameters(conflict=.1,votes=2),
    'retrieval_order': parameters(order='retrieval_score'),
}
METHODS=('baseline',*VARIANTS)
RECIPE=dict(methods=METHODS,variants=VARIANTS,seeds=[42,123,2026],
    features='single_phase_hard_center_GMP_rho01',
    insertion='background_only_after_original_foreground_NMS',
    conflict='intersection_with_original_other_class_foreground / original_candidate_area',
    votes='same_predicted_class_pixel_support_from_distinct_gate_passing_extra_masks_including_self',
    novelty='surviving_unoccupied_pixels_after_vote / original_candidate_area',
    scheduling='fixed_48_cells_no_adaptive_search')


def rescue_rows(strong,extras,params,stats=None):
    stats=stats if stats is not None else {}
    stats.update(candidates=len(extras),gated=0,conflict_rejected=0,accepted=0,added_pixels=0)
    if not strong and not extras:return [],0
    shape=(strong or extras)[0]['mask'].shape
    occupied=np.zeros(shape,bool); class_masks={}
    for row in [*strong,*extras]:
        if row['mask'].dtype!=bool or row['mask'].shape!=shape:raise ValueError('Invalid mask')
        if any(not math.isfinite(float(row[k])) for k in ('score','proposal_score','retrieval_score','retrieval_margin')):
            raise ValueError('Non-finite candidate scores')
    for row in sorted(strong,key=lambda r:-r['score']):
        if row['cls_name']!='background':
            visible=row['mask'] & ~occupied
            occupied |= row['mask']
            class_masks.setdefault(row['cls_name'],np.zeros(shape,bool))[:] |= visible
    eligible=[];seen=set()
    for row in extras:
        if row['cls_name']=='background' or row['proposal_score']<params['proposal_score'] or row['retrieval_score']<params['retrieval_score'] or row['retrieval_margin']<params['margin']:continue
        stats['gated']+=1
        other=np.zeros(shape,bool)
        for cls,mask in class_masks.items():
            if cls!=row['cls_name']:other |= mask
        if (row['mask'] & other).sum()/max(row['mask'].sum(),1)>params['conflict']:
            stats['conflict_rejected']+=1;continue
        digest=queue.mask_hashes([row['mask']])[0]
        if digest in seen:raise ValueError('Duplicate extra mask')
        seen.add(digest);eligible.append(row)
    votes={}
    if params['votes']>1:
        for row in eligible:
            votes.setdefault(row['cls_name'],np.zeros(shape,np.uint32))[:] += row['mask']
    result=list(strong);accepted=0
    for row in sorted(eligible,key=lambda r:-r[params['order']]):
        mask=row['mask'] & ~occupied
        if params['votes']>1:mask &= votes[row['cls_name']]>=params['votes']
        if not mask.any() or mask.sum()/max(row['mask'].sum(),1)<params['novel_fraction']:continue
        # Keep the original final score: masks are disjoint, so downstream sorting
        # cannot undo the selected allocation of overlapping extra candidates.
        result.append(dict(row,mask=mask));occupied |= mask;accepted+=1
        stats['added_pixels']+=int(mask.sum())
    stats['accepted']=accepted
    return result,accepted


def verify_prior(folder,previous,cleanup,common,records):
    manifest=ab.read(folder/'manifest.json'); summary=ab.read(folder/'summary.json')
    if manifest['protocol']!=prior.PROTOCOL or ab.read(folder/'status.json')['status']!='complete':
        raise ValueError('Need a completed overnight rescue source')
    if manifest['recipe']!=old.canonical(prior.RECIPE) or Path(manifest['previous']).resolve()!=previous.resolve():
        raise ValueError('Different rescue recipe or model source')
    if manifest['cleanup_recipe']!=cleanup or summary['evaluation_ids']!=common['val_ids']:
        raise ValueError('Different cleanup or evaluation IDs')
    gmp.verify(manifest['code'],'Prior source');gmp.verify(manifest['references'],'Prior inputs')
    records.extend(manifest['code']);records.extend(manifest['references'])
    records.extend(ab.identities([folder/n for n in ('manifest.json','status.json','summary.json')]))
    refs={}
    for method in ('baseline','rescue_strict'):
        for seed in old.SEEDS:
            key=f'{method}_{seed}'
            refs[key]=queue.verified_stage(folder,key,records)
            records.extend(ab.identities([folder/key/'complete.json']))
    cache=ROOT/'storage/raw_banks/voc'/folder.name/'rescue'
    count=0
    for stem in common['val_ids']:
        marker=cache/stem/'complete.json';value=ab.read(marker)
        gmp.verify(value['artifacts'],'Prior extra cache')
        records.extend(value['artifacts']);records.extend(ab.identities([marker]))
        count+=value['result']['extra_count']
    return cache,refs,count


def evaluation_plan():
    return [[(m,s) for s in old.SEEDS] for m in METHODS]


def added_pixel_counts(gt,original,prediction):
    added=(original==0)&(prediction>0)&(gt!=255)
    def counts(selection):
        return np.bincount(prediction[selection].astype(np.int64),minlength=21).tolist()
    return dict(added_background_by_class=counts(added&(gt==0)),
                recovered_by_class=counts(added&(gt==prediction)),
                new_wrong_foreground_by_class=counts(added&(gt>0)&(gt!=prediction)))


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
        if job['method']!='baseline':
            with np.load(job['baseline_per_image'],allow_pickle=False) as data:
                reference_matrices={str(i):cm.copy() for i,cm in zip(data['ids'],data['confusions'])}
        expanded=job['method']!='baseline'
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
            if added.shape!=(len(eh),1280) or not np.isfinite(added).all():raise ValueError('Invalid cached vectors')
            base=read_vectors(folder,identity,group,queue.mask_hashes(strong))
            return np.concatenate([base,added],axis=0)
        def nms_hook(rows,threshold):
            if job['method']=='baseline':return nms(rows,threshold)
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
            stats={};result,accepted=rescue_rows(kept,extra,VARIANTS[job['method']],stats);state['accepted']=accepted;state['gate_stats']=stats
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
                foreground_fn=int(((gt>0)&valid&(pred==0)).sum()),accepted_rescues=state['accepted'],gate_stats=state.get('gate_stats',{}),
                **added_pixel_counts(gt,state.get('strong_prediction',pred),pred)))
            return pred
        inference.load_strict_proposals=load_hook;queue.cache_vectors=vectors_hook
        inf.nms_masks_per_class=nms_hook;inference.rasterize_instances=raster_hook
        job['_report']=(ids,cms,counts,['background']+list(inf.DATASET_CLASSES))
    queue.install_cached_eval=hook


def worker(job):
    import torch
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();started=time.monotonic();status='failed';error=None;category='global'
    try:
        if job['stage']!='eval':raise ValueError('This batch permits cached evaluation only')
        else:
            install(job);old.previous.worker(job)
            ids,cms,counts,classes=job.pop('_report')
            if ids!=job['val_ids']:raise ValueError('Incomplete evaluation')
            ab.write(Path(job['attempt'])/'result.json',dict(method=job['method'],seed=job['seed'],classes=classes,
                metrics=oracle.metrics(np.sum(cms,axis=0)),images=counts,
                effective_candidate_pool='original_clean' if job['method']=='baseline' else 'original_clean_union_relaxed_score0_deduplicated_by_mask',
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
    p.add_argument('--rescue-previous',type=Path,default=Path('experiments/overnight_rescue_20261004_005647'))
    p.add_argument('--version',default='rescue_mechanisms_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
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
    cache,prior_refs,extra_count=verify_prior(args.rescue_previous.resolve(),args.previous,meta['config'],common,records)
    code=ab.source_identity();identity=old.canonical(dict(protocol=PROTOCOL,recipe=RECIPE,code=code,references=records,
        previous=str(args.previous.resolve()),rescue_previous=str(args.rescue_previous.resolve()),gpus=args.gpus,cleanup_recipe=meta['config']))
    output=ROOT/'experiments'/args.version;logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version mismatch or --resume missing')
    elif logs.exists():raise FileExistsError('Version already used')
    stages=ab.Stages(output,True);stages.verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+1*2**30:raise queue.Deferred('Need 1 GiB allowance + 5 GiB reserve')
    print('[Plan] 48 fixed evaluations; baseline + strict historical replay first; verified existing extra cache; NO training or SAM3 extraction.',flush=True)
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
    entry=queue.__file__;queue.__file__=__file__;status='running';error=None;expanded_count=clean['count']+extra_count
    # Preserve all completed reports even when a resumed budget expires before
    # the scheduler gets to that stage again. Stages.verify_all ran above.
    for method in METHODS:
        for seed in old.SEEDS:
            key=f'{method}_{seed}';marker=output/key/'complete.json'
            if marker.exists():reports[key]=ab.read(marker)['result']
    def flush():
        values={k:ab.read(v['report']) for k,v in reports.items()};summary={}
        for method in METHODS:
            rows={str(s):values[f'{method}_{s}'] for s in old.SEEDS if f'{method}_{s}' in values}
            scores=[r['metrics']['mIoU'] for r in rows.values()]
            summary[method]=dict(complete=len(rows)==3,seeds={s:dict(metrics=r['metrics'],
                background_fp=sum(x['background_fp'] for x in r['images']),foreground_fn=sum(x['foreground_fn'] for x in r['images']),
                accepted_rescues=sum(x['accepted_rescues'] for x in r['images']),
                gate_totals={k:sum(x.get('gate_stats',{}).get(k,0) for x in r['images']) for k in ('candidates','gated','conflict_rejected','accepted','added_pixels')},
                added_pixel_totals={k:np.sum([x[k] for x in r['images']],axis=0).tolist() for k in ('added_background_by_class','recovered_by_class','new_wrong_foreground_by_class')},
                strict_delta_pp=100*(r['metrics']['mIoU']-values[f'rescue_strict_{s}']['metrics']['mIoU']) if f'rescue_strict_{s}' in values else None,
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
                OVERRIDE_INF_CENTROID_WEIGHT='0.4')
            log=logs/(key+'_'+attempt.name+'.log');queue.launch(job,attempt,env,log,cancel,deadline)
            artifacts=[attempt,log];result={}
            if stage=='eval':
                metric=ab.read(job['metrics']);ref=old.canonical(ab.read(model['metrics']))
                count=clean['count'] if method=='baseline' else expanded_count
                # Explicitly declared pool/scoring differences; all other settings still strict.
                ref['proposal_count_before']=ref['proposal_count_after']=count
                old.validate_eval(metric,ref,common['val_ids'],count,.2)
                old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
                if metric['feature_signature']!=ab.read(Path(model['bank'])/'meta/feature_signature.json'):raise ValueError('Adapter/bank signature differs')
                detail=ab.read(attempt/'result.json')
                if detail['metrics']['confusion_matrix']!=metric['confusion_matrix']:raise ValueError('Metric mismatch')
                if method=='baseline':
                    old.mpsa.require_baseline_replay(ab.read(model['metrics']),metric)
                    old.require_image_replay(model['per_image'],str(attempt/'per_image_confusion.npz'))
                if method in ('baseline','rescue_strict'):
                    previous_result=prior_refs[f'{method}_{seed}']
                    if detail['metrics']['confusion_matrix']!=ab.read(previous_result['report'])['metrics']['confusion_matrix']:raise ValueError('Previous rescue result differs')
                    old.require_image_replay(previous_result['per_image'],str(attempt/'per_image_confusion.npz'))
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
        flush();ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,logs])),read_only_cache=str(cache),automatic_deletion=False))
        plan=evaluation_plan()
        for batch in plan[:2]:queue.run_round(batch,args.gpus,evaluate,cancel,deadline,fatal)
        queue.run_round([item for batch in plan[2:] for item in batch],args.gpus,evaluate,cancel,deadline,local_failure)
        gmp.verify(records,'Historical inputs')
        if ab.source_identity()!=code:raise ValueError('Source changed')
        status='complete' if len(reports)==len(METHODS)*len(old.SEEDS) else 'partial'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except KeyboardInterrupt as exc:status='interrupted';error=str(exc);cancel.set();raise
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();queue.__file__=entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink();notify_experiment(args.version,status,str(output),time.time()-ab.read(budget)['started_at'])


if __name__=='__main__':main()
