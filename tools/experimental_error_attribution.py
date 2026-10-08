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
PROTOCOL = 'voc_error_attribution_v2'
NOTIFY_SHA256 = '7bf4e739a4b7cf15024d1c2a6eda8aecc524afc20d340a4ef3a8cb5332615c07'


def checked_history(previous, clean):
    """Allow exactly the reviewed mail helper as a new, non-inference utility.

    Historical file hashes still use the original verifier. The temporary view is
    restored before recording the new manifest, which includes the helper hash.
    """
    helper=(ROOT/'project_utils/experiment_notify.py').resolve()
    actual=hashlib.sha256(helper.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if actual!=NOTIFY_SHA256:
        raise ValueError('Email helper differs from reviewed version; upload matching experiment_notify.py')
    source_identity=ab.source_identity
    try:
        ab.source_identity=lambda:[r for r in source_identity() if Path(r['path']).resolve()!=helper]
        return old.checked_history(previous,clean)
    finally:
        ab.source_identity=source_identity



def boundary_band(gt, radius=3):
    """Valid pixels near direct transitions OR an ignore strip; not true boundary IoU."""
    from scipy.ndimage import binary_dilation
    return (base.boundary_band(gt,radius) | binary_dilation(gt==255,iterations=radius)) & (gt!=255)


def split_oracles(gt, prediction, rows, classes):
    mapping={c:i+1 for i,c in enumerate(classes)}
    clear=prediction.copy();clear[gt==0]=0
    conflict=prediction.copy(); recover=prediction.copy()
    for r in rows:
        cls=mapping.get(r['cls_name'],0)
        if not cls:continue
        correct=r['mask'] & (gt==cls)
        conflict[correct & (prediction>0)]=cls
        recover[correct & (prediction==0)]=cls
    both=base.pixel_oracle([r['mask'] for r in rows],[mapping.get(r['cls_name'],0) for r in rows],gt,prediction)
    return dict(background_clear=clear,foreground_conflict=conflict,foreground_recover=recover,pixel_combined=both)


LOSS_NAMES=('no_raw_coverage','cleanup_coverage_lost','invalid_features','no_correct_class',
            'margin_filtered','nms_filtered','composition_lost')


def coverage(masks, shape):
    out=np.zeros(shape,bool)
    for mask in masks:out |= mask
    return out


def attribute(gt,prediction,raw,masks,valid_ids,predictions,margin,after,classes):
    """Partition foreground FN by first absent coverage; descriptive, not causal proof."""
    if len(valid_ids)!=len(predictions) or len(set(valid_ids))!=len(valid_ids):raise ValueError('Retrieval alignment error')
    mapping={c:i+1 for i,c in enumerate(classes)}
    raw_cover=coverage(raw,gt.shape);clean_cover=coverage(masks,gt.shape)
    valid_cover=coverage([masks[i] for i in valid_ids],gt.shape)
    result={}
    for cls in range(1,21):
        target=(gt==cls)&(prediction==0);remaining=target.copy();counts={}
        correct=[masks[i] for i,r in zip(valid_ids,predictions) if mapping.get(r['label'],0)==cls]
        passed=[masks[i] for i,r in zip(valid_ids,predictions) if mapping.get(r['label'],0)==cls and r['margin']>=margin]
        kept=[r['mask'] for r in after if mapping.get(r['cls_name'],0)==cls]
        covers=[raw_cover,clean_cover,valid_cover,coverage(correct,gt.shape),coverage(passed,gt.shape),coverage(kept,gt.shape)]
        for name,cov in zip(LOSS_NAMES,covers):
            counts[name]=int((remaining & ~cov).sum());remaining &= cov
        counts['composition_lost']=int(remaining.sum())
        counts['total_fn']=int(target.sum())
        if sum(counts[k] for k in LOSS_NAMES)!=counts['total_fn']:raise ValueError('FN partition mismatch')
        result[classes[cls-1]]=counts
    return result


def aggregate_attribution(images):
    first=next(iter(images.values()))['attribution']
    return {c:{k:sum(v['attribution'][c][k] for v in images.values()) for k in row} for c,row in first.items()}


def install(job):
    original=queue.install_cached_eval
    def hook(j):
        original(j)
        import run_inference as inference
        import src.inference as inf
        import src.retrieval as retrieval
        from PIL import Image
        state={};collected={};details={}
        load=inference.load_strict_proposals;raster=inference.rasterize_instances
        build=inference.build_feature_extractor; classify=inf.classify_queries; search=retrieval._search
        def load_hook(directory,stem,height,width):
            masks,scores=load(directory,stem,height,width)
            state.clear();state.update(stem=stem,masks=masks,scores=scores,valid_ids=[],predictions=[],neighbors=[])
            with np.load(Path(directory)/(stem+'.npz'),allow_pickle=False) as data:
                state['source_indices']=data['source_indices'].tolist()
            return masks,scores
        def build_hook(*args,**kwargs):
            extractor=build(*args,**kwargs);extract=extractor.extract_features_batch
            def record(rgb,masks):
                vectors=extract(rgb,masks)
                state['valid_ids']=[i for i,v in enumerate(vectors) if np.isfinite(v).all() and np.linalg.norm(v)>=1e-6]
                return vectors
            extractor.extract_features_batch=record
            return extractor
        def search_hook(*args,**kwargs):
            result=search(*args,**kwargs)
            state['search']=(result[0].copy(),result[1].copy())
            return result
        def classify_hook(*args,**kwargs):
            result=classify(*args,**kwargs)
            state['predictions']=[vars(r).copy() for r in result]
            sims,indices=state.pop('search');labels=kwargs['all_labels']
            state['neighbors']=[[dict(bank_row=int(i),label=str(labels[i]),similarity=float(s))
                for s,i in zip(ss[:10],ii[:10]) if i>=0] for ss,ii in zip(sims,indices)]
            return result
        def raster_hook(rows,height,width):
            queue.ensure_resources(float('inf'))
            pred=raster(rows,height,width);stem=state['stem']
            if stem in details:raise ValueError('Duplicate image')
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im:gt=np.array(im)
            with np.load(Path(job['raw_proposals'])/(stem+'.npz'),allow_pickle=False) as data:
                raw=[m.astype(bool) for m in data['masks']]
            if any(m.shape!=gt.shape for m in raw):raise ValueError('Raw proposal shape mismatch')
            cms,detail=base.diagnose(gt,state['masks'],state['scores'],rows,inf.DATASET_CLASSES,pred)
            # Keep original region arrays for stratification, replace the misleading boundary metric.
            cms.pop('baseline_boundary');cms.pop('baseline_interior')
            band=boundary_band(gt);valid=gt!=255;errors=(gt!=pred)&valid
            cms['baseline_boundary_proxy']=base.confusion(gt,pred,band)
            cms['baseline_interior_proxy']=base.confusion(gt,pred,valid&~band)
            detail['counts'].update(boundary_pixels=int(band.sum()),boundary_errors=int((errors&band).sum()),
                interior_pixels=int((valid&~band).sum()),interior_errors=int((errors&~band).sum()))
            for name,value in split_oracles(gt,pred,rows,inf.DATASET_CLASSES).items():cms[name]=base.confusion(gt,value)
            detail['attribution']=attribute(gt,pred,raw,state['masks'],state['valid_ids'],state['predictions'],
                inf.INF_RETRIEVAL_MARGIN_THR,rows,inf.DATASET_CLASSES)
            # All candidates recorded; target regions can be inspected without selecting by score.
            candidates=[]
            lookup={i:(r,n) for i,r,n in zip(state['valid_ids'],state['predictions'],state['neighbors'])}
            for i,m in enumerate(state['masks']):
                r,n=lookup.get(i,(None,[]))
                hist=np.bincount(gt[m & valid],minlength=21).tolist()
                candidates.append(dict(index=i,source_index=state['source_indices'][i],mask_sha256=queue.mask_hashes([m])[0],proposal_score=float(state['scores'][i]),
                    gt_histogram=hist,prediction=r,neighbors=n))
            detail['candidates']=candidates
            # Bounded visual evidence: one image panel per image with chair pixels assigned to sofa.
            chair=inf.DATASET_CLASSES.index('chair')+1;sofa=inf.DATASET_CLASSES.index('sofa')+1
            if np.any((gt==chair)&(pred==sofa)):
                folder=Path(job['attempt'])/'chair_sofa';folder.mkdir(exist_ok=True)
                with Image.open(job['images'][stem]) as im:rgb=np.array(im.convert('RGB'))
                mark=rgb.copy();mark[(gt==chair)&(pred==sofa)]=[255,0,255]
                Image.fromarray(np.concatenate([rgb,mark],axis=1)).save(folder/(stem+'.jpg'))
            details[stem]=detail
            for name,cm in cms.items():collected.setdefault(name,[]).append(cm)
            if len(details)%50==0:print(f'[Attribution] {len(details)}/1449',flush=True)
            return pred
        inference.load_strict_proposals=load_hook;inference.build_feature_extractor=build_hook
        inference.rasterize_instances=raster_hook;inf.classify_queries=classify_hook;retrieval._search=search_hook
        job['_collected']=collected;job['_details']=details;job['_classes']=list(inf.DATASET_CLASSES)
    queue.install_cached_eval=hook


def worker(job):
    base.install=install
    base.worker(job)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    p.add_argument('--clean-proposals-dir',type=Path,default=Path(old.mpsa.DEFAULT_CLEAN))
    p.add_argument('--version',default='error_attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
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
        clean=str(args.clean_proposals_dir.resolve()),seeds=list(old.SEEDS),boundary_radius=3,component_connectivity=4,attribution='ordered_first_loss_v1'))
    output=ROOT/'experiments'/args.version;logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version differs or --resume missing')
    elif logs.exists():raise FileExistsError(logs)
    stages=ab.Stages(output,True);stages.verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+2**30:raise queue.Deferred('Need 1 GiB allowance + 5 GiB reserve')
    print('[Plan] Three historical model replays + GT-only diagnostics; no training/SAM3/cleaning.',flush=True)
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
        queue.run_round(list(old.SEEDS),args.gpus,execute,cancel,deadline,lambda item,exc:(_ for _ in ()).throw(exc))
        gmp.verify(records,'Historical inputs')
        if ab.source_identity()!=code:raise ValueError('Code changed during experiment')
        results={s:ab.read(r['diagnostics']) for s,r in reports.items()}
        error_summary={}
        for seed,report in results.items():
            images=list(report['images'].values())
            counts={k:sum(im['counts'][k] for im in images) for k in images[0]['counts']}
            ious=np.asarray([v for im in images for v in im['region_iou']])
            cls=np.asarray([v for im in images for v in im['region_class']])
            areas=np.asarray([v for im in images for v in im['region_area']])
            def region_stats(values):
                return dict(count=len(values),mean_best_iou=float(values.mean()) if len(values) else None,
                    recall_at_05=float((values>=.5).mean()) if len(values) else None,
                    recall_at_075=float((values>=.75).mean()) if len(values) else None)
            error_summary[seed]=dict(pixel_counts=counts,regions=region_stats(ious),
                regions_by_class={report['classes'][c]:region_stats(ious[cls==c]) for c in range(1,21)},
                region_area_strata={str(threshold):dict(region_stats(ious[areas>=threshold]),
                    foreground_area_fraction=float(areas[areas>=threshold].sum()/max(areas.sum(),1))) for threshold in (1,100,300,1000)},
                area_weighted_recall_05=float((areas*(ious>=.5)).sum()/max(areas.sum(),1)))
        summary={}
        for name in results['42']['results']:
            values=[results[str(s)]['results'][name]['mIoU'] for s in old.SEEDS]
            summary[name]=dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),
                by_seed={str(s):v for s,v in zip(old.SEEDS,values)},
                delta_pp_by_seed=None if 'boundary' in name or 'interior' in name else {str(s):100*(v-results[str(s)]['results']['baseline']['mIoU']) for s,v in zip(old.SEEDS,values)})
        ab.write(output/'summary.json',dict(status='complete',protocol=PROTOCOL,metrics=summary,reports=reports,
            error_breakdown=error_summary,classes=results['42']['classes'],attribution={s:aggregate_attribution(r['images']) for s,r in results.items()},
            evaluation_ids=common['val_ids'],note='GT-assisted diagnostics; differences are not additive or deployable gains. SAM3 direct unavailable without class provenance.'))
        lines=['GT-assisted diagnostics; NOT deployable model improvements.','', '| Mode | Mean mIoU % | Seed SD pp |','|---|---:|---:|']
        lines += [f'| {k} | {100*v["mean"]:.4f} | {100*v["sample_std"]:.4f} |' for k,v in summary.items()]
        (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status='complete'
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
