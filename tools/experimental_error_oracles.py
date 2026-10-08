#!/usr/bin/env python3
"""Read-only VOC error diagnosis. GT-assisted outputs are NOT deployable scores."""
from __future__ import annotations
import argparse
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
ab, queue, gmp = old.ab, old.queue, old.gmp
PROTOCOL = 'voc_error_oracles_v1'


def confusion(gt, pred, subset=None):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape or not np.isin(gt, list(range(21))+[255]).all():
        raise ValueError('Invalid VOC ground truth/shape')
    if not np.isin(pred, range(21)).all(): raise ValueError('Invalid prediction')
    valid = gt != 255
    if subset is not None: valid &= subset
    return np.bincount((21*gt[valid].astype(np.int64)+pred[valid]).ravel(), minlength=441).reshape(21,21)


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    union = cm.sum(0)+cm.sum(1)-cm.diagonal()
    iou = np.divide(cm.diagonal(), union, out=np.full(21,np.nan), where=union>0)
    return dict(mIoU=float(np.nanmean(iou)), per_class_iou=[None if not np.isfinite(x) else float(x) for x in iou],
                confusion_matrix=cm.tolist())


def compose(masks, labels, shape):
    result = np.zeros(shape, np.uint8)
    for mask, label in zip(masks, labels):
        if label: result[mask & (result==0)] = label
    return result


def majority(mask, gt):
    values = gt[mask & (gt!=255)]
    return int(np.bincount(values, minlength=21).argmax()) if len(values) else 0


def pixel_oracle(masks, labels, gt, fallback):
    """Best available class per pixel; does not claim a realizable global ordering."""
    result = fallback.copy()
    for mask, label in zip(masks, labels):
        result[mask & (gt==label)] = label
    # Background is always available by rejecting foreground coverage.
    result[gt==0] = 0
    return result


def boundary_band(gt, radius=3):
    from scipy.ndimage import binary_dilation
    valid = gt!=255
    edge = np.zeros(gt.shape,bool)
    diff = (gt[1:]!=gt[:-1]) & valid[1:] & valid[:-1]
    edge[1:] |= diff; edge[:-1] |= diff
    diff = (gt[:,1:]!=gt[:,:-1]) & valid[:,1:] & valid[:,:-1]
    edge[:,1:] |= diff; edge[:,:-1] |= diff
    return binary_dilation(edge, iterations=radius) & valid


def diagnose(gt, masks, scores, rows, classes, prediction):
    from scipy.ndimage import label as components
    if len(masks)!=len(scores) or not np.isfinite(scores).all(): raise ValueError('Bad proposals')
    if any(m.shape!=gt.shape or m.dtype!=bool for m in masks): raise ValueError('Bad masks')
    mapping = {c:i+1 for i,c in enumerate(classes)}
    rm = [r['mask'] for r in rows]; rl = [mapping.get(r['cls_name'],0) for r in rows]
    if not np.array_equal(compose(rm,rl,gt.shape), prediction): raise ValueError('Rasterizer mismatch')
    outputs = {'baseline':prediction}
    # Preserve selected masks, scores/order and NMS membership: change class only.
    outputs['class_selected'] = compose(rm,[majority(m,gt) for m in rm],gt.shape)
    outputs['pixel_selected'] = pixel_oracle(rm,rl,gt,prediction)
    order = np.argsort(-np.asarray(scores),kind='stable')
    ordered = [masks[i] for i in order]
    oracle_labels = [majority(m,gt) for m in ordered]
    raw_class = compose(ordered,oracle_labels,gt.shape)
    outputs['class_all_proposals'] = raw_class
    outputs['class_and_pixel_all'] = pixel_oracle(ordered,oracle_labels,gt,raw_class)
    selected_masks=[]; selected_labels=[]; ious=[]; region_classes=[]; areas=[]
    valid=gt!=255
    # Semantic GT connected components are a region proxy, not instance GT.
    for cls in range(1,21):
        cc,n=components(gt==cls)  # fixed 4-connectivity
        for index in range(1,n+1):
            region=cc==index; area=int(region.sum())
            overlaps=[]
            for mask in masks:
                intersection=int((mask & region).sum())
                union=int((mask & valid).sum())+area-intersection
                overlaps.append(intersection/union if union else 0.)
            best=int(np.argmax(overlaps)) if overlaps else None
            iou=overlaps[best] if best is not None else 0.
            ious.append(iou); region_classes.append(cls); areas.append(area)
            if iou>0: selected_masks.append(masks[best]); selected_labels.append(cls)
    # Fixed class/component order. Also report optimistic pixel assignment separately.
    best_map=compose(selected_masks,selected_labels,gt.shape)
    outputs['best_region_fixed_order']=best_map
    outputs['best_region_pixel']=pixel_oracle(selected_masks,selected_labels,gt,best_map)
    band=boundary_band(gt)
    errors=(prediction!=gt)&valid
    counts=dict(valid=int(valid.sum()), errors=int(errors.sum()),
                foreground_fp=int(((gt==0)&(prediction>0)).sum()),
                foreground_fn=int(((gt>0)&valid&(prediction==0)).sum()),
                foreground_wrong_class=int(((gt>0)&valid&(prediction>0)&errors).sum()),
                boundary_pixels=int(band.sum()), boundary_errors=int((errors&band).sum()),
                interior_pixels=int((valid&~band).sum()), interior_errors=int((errors&~band).sum()))
    matrices={k:confusion(gt,v) for k,v in outputs.items()}
    matrices['baseline_boundary']=confusion(gt,prediction,band)
    matrices['baseline_interior']=confusion(gt,prediction,valid&~band)
    return matrices,dict(counts=counts,region_iou=ious,region_class=region_classes,region_area=areas)


def install(job):
    original = queue.install_cached_eval
    def hook(j):
        original(j)
        import run_inference as inference
        import src.inference as inf
        from PIL import Image
        load=inference.load_strict_proposals; raster=inference.rasterize_instances
        state={}; collected={}; details={}
        def load_hook(directory,stem,height,width):
            masks,scores=load(directory,stem,height,width)
            state.update(stem=stem,masks=masks,scores=scores)
            return masks,scores
        def raster_hook(rows,height,width):
            queue.ensure_resources(float('inf'))
            result=raster(rows,height,width)
            stem=state['stem']
            if stem in details: raise ValueError('Duplicate evaluated ID')
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im: gt=np.array(im)
            matrices,detail=diagnose(gt,state['masks'],state['scores'],rows,inf.DATASET_CLASSES,result)
            details[stem]=detail
            for name,cm in matrices.items(): collected.setdefault(name,[]).append(cm)
            if len(details)%50==0: print(f'[Diagnosis] {len(details)}/1449',flush=True)
            return result
        inference.load_strict_proposals=load_hook; inference.rasterize_instances=raster_hook
        job['_collected']=collected; job['_details']=details; job['_classes']=list(inf.DATASET_CLASSES)
    queue.install_cached_eval=hook


def worker(job):
    install(job)
    old.previous.worker(job)
    details=job.pop('_details'); collected=job.pop('_collected'); classes=job.pop('_classes')
    ids=list(details)
    if ids!=job['val_ids'] or len(ids)!=1449: raise ValueError('Incomplete/incorrect evaluation order')
    dest=Path(job['attempt'])
    np.savez_compressed(dest/'oracle_per_image.npz',ids=np.asarray(ids),**{k:np.stack(v) for k,v in collected.items()})
    report={k:metrics(np.sum(v,axis=0)) for k,v in collected.items()}
    ab.write(dest/'diagnostics.json',dict(classes=['background']+classes,results=report,images=details,
        sam3_direct=dict(status='unavailable',reason='Historical cache API preserves masks/scores, not verified original prompt classes'),
        note='GT-assisted development diagnostics, not deployable results. Region=4-connected semantic component; pixel oracle is not a global mask ordering.'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    p.add_argument('--clean-proposals-dir',type=Path,default=Path(old.mpsa.DEFAULT_CLEAN))
    p.add_argument('--version',default='error_oracles_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1]);p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or not np.isfinite(args.max_hours) or args.max_hours<=0: p.error('Invalid devices/budget')
    for gpu in args.gpus: ab.check_gpu(gpu)
    common,refs,split,records,clean,raw,sig=old.checked_history(args.previous.resolve(),args.clean_proposals_dir.resolve())
    if len(common['val_ids'])!=1449: raise ValueError('Expected full VOC val')
    code=ab.source_identity()
    identity=old.canonical(dict(protocol=PROTOCOL,previous=str(args.previous.resolve()),references=records,code=code,gpus=args.gpus,
        clean=str(args.clean_proposals_dir.resolve()),seeds=list(old.SEEDS),boundary_radius=3,component_connectivity=4))
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
                     weights=model['weights'],bank=model['bank'],cleanup='PRE_CLEANED',attempt=str(attempt),metrics=str(attempt/'metrics.json'))
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
            def region_stats(values):
                return dict(count=len(values),mean_best_iou=float(values.mean()) if len(values) else None,
                    recall_at_05=float((values>=.5).mean()) if len(values) else None,
                    recall_at_075=float((values>=.75).mean()) if len(values) else None)
            error_summary[seed]=dict(pixel_counts=counts,regions=region_stats(ious),
                regions_by_class={report['classes'][c]:region_stats(ious[cls==c]) for c in range(1,21)})
        summary={}
        for name in results['42']['results']:
            values=[results[str(s)]['results'][name]['mIoU'] for s in old.SEEDS]
            summary[name]=dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),
                by_seed={str(s):v for s,v in zip(old.SEEDS,values)},
                delta_pp_by_seed={str(s):100*(v-results[str(s)]['results']['baseline']['mIoU']) for s,v in zip(old.SEEDS,values)})
        ab.write(output/'summary.json',dict(status='complete',protocol=PROTOCOL,metrics=summary,reports=reports,
            error_breakdown=error_summary,classes=results['42']['classes'],
            evaluation_ids=common['val_ids'],note='GT-assisted diagnostics; differences are not additive or deployable gains. SAM3 direct unavailable without class provenance.'))
        lines=['GT-assisted diagnostics; NOT deployable model improvements.','', '| Mode | Mean mIoU % | Seed SD pp |','|---|---:|---:|']
        lines += [f'| {k} | {100*v["mean"]:.4f} | {100*v["sample_std"]:.4f} |' for k,v in summary.items()]
        (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status='complete'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        ab.write(output/'status.json',dict(status=status,error=error,completed=sorted(reports)))
        queue.__file__=entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink()
    print(f'[{status}] {output}',flush=True)


if __name__=='__main__':main()
