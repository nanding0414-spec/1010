#!/usr/bin/env python3
"""Fixed patch-prototype foreground verification; no new MLP training."""
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
PROTOCOL='patch_foreground_verifier_v1'
RECIPE=dict(images_per_class=16,patches_per_image_class=32,max_patches_per_class=2048,
    core_radius=7,core_coverage=.95,foreground_prototypes=4,background_prototypes=[1,8],
    temperature=.1,threshold=-.05,kmeans_iterations=20,seed=42,
    feature_space='SAM3_processed_L2_tokens_no_region_adapter',grid=[72,72],
    scoring='temperature_logmeanexp_fg_minus_bg',resize='bilinear_align_corners_false',
    placement='after_classification_before_class_NMS',empty='drop_no_fallback')


def unit(x):
    x=np.asarray(x,np.float32)
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or (norm<1e-8).any():raise ValueError('Nonfinite/zero prototype input')
    return x/norm


def choose_images(plan,split):
    allowed=set(split['train_images']);selected=set()
    if allowed & set(split['validation_images']):raise ValueError('Split overlap')
    classes=sorted({r['cls'] for rows in plan.values() for r in rows if r['cls']!='background'})
    if len(classes)!=20:raise ValueError('Need 20 foreground classes')
    for cls in classes:
        ids=[s for s in allowed if any(r['cls']==cls for r in plan.get(s,[]))]
        ids.sort(key=lambda s:hashlib.sha256(('42:'+s).encode()).hexdigest())
        if not ids:raise ValueError('No training images for '+cls)
        selected.update(ids[:RECIPE['images_per_class']])
    return sorted(selected)


def core_indices(seg):
    import cv2
    from scipy.ndimage import binary_erosion
    if seg.ndim!=2 or not np.isin(seg,list(range(21))+[255]).all():raise ValueError('Invalid pseudo PNG')
    resized=cv2.resize(seg.astype(np.uint8),(1008,1008),interpolation=cv2.INTER_NEAREST)
    result={}
    for c in range(21):
        core=binary_erosion(resized==c,iterations=RECIPE['core_radius'],border_value=0)
        cover=cv2.resize(core.astype(np.float32),(72,72),interpolation=cv2.INTER_AREA)
        result[c]=np.flatnonzero(cover.ravel()>=RECIPE['core_coverage'])
    return result


def cluster(x,k):
    x=unit(x)
    if len(x)<k:raise ValueError(f'Insufficient core patches for {k} prototypes')
    # Deterministic farthest-first initialization, spherical Lloyd iterations.
    centers=[x[0]];distance=x@centers[0]
    for _ in range(1,k):
        centers.append(x[int(np.argmin(distance))]);distance=np.maximum(distance,x@centers[-1])
    centers=np.stack(centers)
    for _ in range(RECIPE['kmeans_iterations']):
        labels=np.argmax(x@centers.T,axis=1);new=centers.copy()
        for i in range(k):
            members=x[labels==i]
            if len(members):new[i]=unit(members.mean(0))
        centers=new
    return centers


def evidence(tokens,foreground,background):
    import torch
    import torch.nn.functional as F
    if tokens.ndim!=2 or tokens.shape[0]!=5184 or not torch.isfinite(tokens).all() or (tokens.norm(dim=-1)<1e-8).any():
        raise ValueError('Invalid token grid')
    if np.asarray(foreground).shape[:2]!=(20,4) or np.asarray(foreground).shape[-1]!=tokens.shape[-1]:raise ValueError('Invalid foreground prototypes')
    foreground=unit(foreground);background=unit(background)
    tokens=F.normalize(tokens.float(),dim=-1)
    fg=torch.as_tensor(foreground,device=tokens.device,dtype=torch.float32)
    bg=torch.as_tensor(background,device=tokens.device,dtype=torch.float32)
    tau=RECIPE['temperature']
    fs=(tokens@fg.reshape(-1,fg.shape[-1]).T).reshape(len(tokens),20,fg.shape[1])
    bs=tokens@bg.T
    # log-mean-exp removes the artificial advantage of a larger prototype count.
    out=tau*(torch.logsumexp(fs/tau,dim=-1)-math.log(fg.shape[1]))
    out-=tau*(torch.logsumexp(bs/tau,dim=-1)-math.log(len(bg)))[:,None]
    if not torch.isfinite(out).all():raise ValueError('Nonfinite evidence')
    return out.T.reshape(20,72,72)


def read_cache(root,kind,stem):
    folder=Path(root)/kind/stem
    value=ab.read(folder/'complete.json')
    gmp.verify(value['artifacts'],'Verifier '+kind+' cache')
    return value['result']['file']


def extraction_worker(job):
    import torch
    from PIL import Image
    sys.path.insert(0,str(ROOT/'sam3-main'))
    ab.install_precision()
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    if ab.digest(job['checkpoint'])!=job['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    model=SAM3FeatureExtractor(model_path=job['checkpoint'],device=torch.device('cuda:0'),resolution=1008,
        feature_level=-1,feature_source='trunk_fpn',token_norm='layernorm',context_subtract=0,
        use_bf16=False,use_interactivity=False,enable_segmentation=False)
    kind='samples' if job['stage']=='sample' else 'evidence'
    stages=ab.Stages(Path(job['new_cache'])/kind,True)
    proto=None
    if kind=='evidence':
        if ab.digest(job['prototypes'])!=job['prototype_sha256']:raise ValueError('Prototype hash mismatch')
        with np.load(job['prototypes'],allow_pickle=False) as data:proto={k:data[k] for k in data.files}
    for stem in job['ids']:
        queue.ensure_resources(job['deadline'])
        def action(attempt):
            image_path=job['images'][stem]
            if ab.digest(image_path)!=job['input_hashes'][stem]['image']:raise ValueError('Image changed')
            with Image.open(image_path) as im:rgb=np.array(im.convert('RGB'))
            inp,_=model._preprocess_image(rgb)
            with torch.inference_mode():tokens,grid=model._forward_tokens(inp)
            if tuple(grid)!=(72,72) or tuple(tokens.shape)!=(1,5184,1280):raise ValueError('Feature geometry changed')
            if not torch.isfinite(tokens).all() or (tokens.norm(dim=-1)<1e-8).any():raise ValueError('Invalid tokens')
            file=attempt/'vectors.npz'
            if kind=='samples':
                path=Path(job['pseudo'])/(stem+'.png')
                if ab.digest(path)!=job['input_hashes'][stem]['mask']:raise ValueError('Pseudo label changed')
                with Image.open(path) as im:seg=np.array(im)
                if seg.shape!=rgb.shape[:2]:raise ValueError('Pseudo image shape mismatch')
                rng=np.random.default_rng(int(hashlib.sha256(('42:'+stem).encode()).hexdigest()[:16],16))
                indices=core_indices(seg);vectors=[];labels=[];patches=[]
                for c,ids in indices.items():
                    ids=np.sort(rng.choice(ids,min(len(ids),RECIPE['patches_per_image_class']),replace=False))
                    if len(ids):
                        vectors.append(tokens[0,torch.as_tensor(ids,device=tokens.device)].cpu().numpy())
                        labels.extend([c]*len(ids));patches.extend(ids.tolist())
                if not vectors:raise ValueError('No reliable core patches: '+stem)
                np.savez_compressed(file,vectors=unit(np.concatenate(vectors)),labels=np.array(labels),patches=np.array(patches))
            else:
                path=Path(job['proposals'])/(stem+'.npz')
                if ab.digest(path)!=job['input_hashes'][stem]['mask']:raise ValueError('Proposals changed')
                # Online FP32 tokens must reproduce existing GMP vectors before use.
                with np.load(path,allow_pickle=False) as data:masks=list(data['masks'].astype(bool))
                cached=queue.cache_vectors(Path(job['cache'])/'val'/stem,job['cache_identity'][stem],'C',queue.mask_hashes(masks))
                for i,mask in enumerate(masks[:2]):
                    index,_=old.mpsa.selection(mask,(0,0))
                    selected=tokens[0,torch.as_tensor(index,device=tokens.device,dtype=torch.long)].float()
                    value,_=gmp.solve(selected,torch.ones(len(index),device=tokens.device),.1)
                    if not np.allclose(value.cpu().numpy(),cached[i],atol=5e-5,rtol=5e-5):raise ValueError('Live tokens do not reproduce GMP cache')
                with torch.inference_mode():
                    values={f'bg{k}':evidence(tokens[0],proto['foreground'],proto[f'background{k}']).cpu().numpy() for k in (1,8)}
                np.savez_compressed(file,**values)
            return dict(file=str(file)),[file]
        stages.run(stem,action)
        print(f'[{kind}] {stem}',flush=True)


def build_prototypes(cache,ids,attempt):
    groups={c:[] for c in range(21)}
    for stem in ids:
        with np.load(read_cache(cache,'samples',stem),allow_pickle=False) as data:
            for c in range(21):groups[c].append(data['vectors'][data['labels']==c])
    rng=np.random.default_rng(42);samples={};counts={}
    for c,values in groups.items():
        x=np.concatenate(values);counts[c]=len(x)
        if len(x)<8:raise ValueError(f'Class {c}: insufficient sampled core tokens ({len(x)})')
        samples[c]=x[np.sort(rng.choice(len(x),min(len(x),RECIPE['max_patches_per_class']),replace=False))]
    file=attempt/'prototypes.npz'
    np.savez_compressed(file,foreground=np.stack([cluster(samples[c],4) for c in range(1,21)]),
        background1=cluster(samples[0],1),background8=cluster(samples[0],8))
    ab.write(attempt/'sampling.json',dict(counts_before_cap=counts,counts_used={c:len(x) for c,x in samples.items()},
        training_images=ids,recipe=RECIPE,note='Pseudo core is a heuristic, not verified ground truth'))
    return dict(file=str(file)),[attempt]


def install_verifier(job):
    original=queue.install_cached_eval
    def hook(j):
        original(j)
        import run_inference as inference
        import src.inference as inf
        import torch
        import torch.nn.functional as F
        from PIL import Image
        load=inference.load_strict_proposals;nms=inf.nms_masks_per_class;raster=inference.rasterize_instances
        state={};cms=[];counts=[];ids=[];classes=['background']+list(inf.DATASET_CLASSES)
        visual_count=0
        def load_hook(directory,stem,height,width):
            value=load(directory,stem,height,width);state.clear();state['stem']=stem;return value
        def nms_hook(rows,threshold):
            nonlocal visual_count
            if job['method']=='baseline':return nms(rows,threshold)
            stem=state['stem']
            with np.load(read_cache(job['new_cache'],'evidence',stem),allow_pickle=False) as data:maps=data[job['method']]
            if maps.shape!=(20,72,72) or not np.isfinite(maps).all():raise ValueError('Invalid evidence cache')
            changed=[];removed=0;dropped=0;visual=None
            for row in rows:
                if row['cls_name']=='background':changed.append(row);continue
                cls=classes.index(row['cls_name'])
                score=F.interpolate(torch.from_numpy(maps[cls-1])[None,None],size=row['mask'].shape,
                    mode='bilinear',align_corners=False)[0,0].numpy()
                mask=row['mask'] & (score>=RECIPE['threshold'])
                if row['cls_name']=='bicycle' and visual_count<20:
                    if visual is None:visual=np.zeros(mask.shape,bool)
                    visual |= row['mask'] & ~mask
                removed+=int(row['mask'].sum()-mask.sum())
                if mask.any():changed.append(dict(row,mask=mask))
                else:dropped+=1
            state.update(removed=removed,dropped=dropped)
            if visual is not None:
                with Image.open(job['images'][stem]) as im:rgb=np.array(im.convert('RGB'))
                marked=rgb.copy();marked[visual]=[255,0,255]
                folder=Path(job['attempt'])/'bicycle_removed';folder.mkdir(exist_ok=True)
                Image.fromarray(np.concatenate([rgb,marked],axis=1)).save(folder/(stem+'.jpg'))
                visual_count+=1
            return nms(changed,threshold)
        def raster_hook(rows,height,width):
            queue.ensure_resources(float('inf'))
            pred=raster(rows,height,width);stem=state['stem']
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im:gt=np.array(im)
            valid=gt!=255;cms.append(oracle.confusion(gt,pred));ids.append(stem)
            counts.append(dict(image=stem,background_fp=int(((gt==0)&(pred>0)).sum()),
                foreground_fn=int(((gt>0)&valid&(pred==0)).sum()),removed_mask_pixels=state.get('removed',0),
                dropped_masks=state.get('dropped',0)))
            return pred
        inference.load_strict_proposals=load_hook;inf.nms_masks_per_class=nms_hook;inference.rasterize_instances=raster_hook
        job['_report']=(ids,cms,counts,classes)
    queue.install_cached_eval=hook


def worker(job):
    import torch
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();started=time.monotonic();status='failed';error=None
    try:
        if job['stage'] in ('sample','maps'):extraction_worker(job)
        else:
            install_verifier(job);old.previous.worker(job)
            ids,cms,counts,classes=job.pop('_report')
            if ids!=job['val_ids']:raise ValueError('Incomplete evaluation')
            ab.write(Path(job['attempt'])/'verifier.json',dict(method=job['method'],recipe=RECIPE,classes=classes,
                metrics=oracle.metrics(np.sum(cms,axis=0)),images=counts))
        torch.cuda.synchronize();status='complete'
    except BaseException as exc:error=str(exc);raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,error=error,category='global',
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    p.add_argument('--clean-proposals-dir',type=Path,default=Path(old.mpsa.DEFAULT_CLEAN))
    p.add_argument('--version',default='fg_verifier_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1]);p.add_argument('--max-hours',type=float,default=48)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv);os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or not math.isfinite(args.max_hours) or args.max_hours<=0:p.error('Invalid GPUs/budget')
    for gpu in args.gpus:ab.check_gpu(gpu)
    common,refs,split,records,clean,raw,sig=checked_history(args.previous.resolve(),args.clean_proposals_dir.resolve())
    source=Path(ab.read(args.previous/'manifest.json')['combo_previous'])
    queue.verified_stage(source,'membership',records);plan=ab.read(source/'members.json')
    selected=choose_images(plan,split)
    if set(selected)&set(common['val_ids']):raise ValueError('Training/validation leakage')
    hashes={}
    for stem in selected+common['val_ids']:
        mask=Path(common['pseudo'])/(stem+'.png') if stem in selected else Path(common['proposals'])/(stem+'.npz')
        hashes[stem]=dict(image=ab.digest(common['images'][stem]),mask=ab.digest(mask))
        records+=ab.identities([common['images'][stem],mask])
    code=ab.source_identity();identity=old.canonical(dict(protocol=PROTOCOL,recipe=RECIPE,selected=selected,
        code=code,references=records,previous=str(args.previous.resolve()),gpus=args.gpus))
    output=ROOT/'experiments'/args.version;cache=ROOT/'storage/raw_banks/voc'/args.version/'foreground_verifier';logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version mismatch or --resume missing')
    elif cache.parent.exists() or logs.exists():raise FileExistsError('Version already used')
    stages=ab.Stages(output,True);stages.verify_all()
    for kind in ('samples','evidence'):ab.Stages(cache/kind,True).verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+3*2**30:raise queue.Deferred('Need 3 GiB allowance + 5 GiB reserve')
    print(f'[Plan] {len(selected)} training images, 1449 query images encoded once; 3 baseline + 6 verifier evaluations.',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists():ab.write(budget,dict(started_at=time.time()))
    deadline=ab.read(budget)['started_at']+args.max_hours*3600;cancel=threading.Event()
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Cancelled')
    for s in handlers:signal.signal(s,stop)
    entry=queue.__file__;queue.__file__=__file__;status='running';error=None;reports={};guard=threading.Lock();proto={}
    def execute(key,stage,gpu,ids=None,method='baseline',seed=42):
        def action(attempt):
            model=refs[seed];job=dict(common,stage=stage,attempt=str(attempt),ids=ids,method=method,seed=seed,
                role='gmp2',group='C',variant='d2',raw_signature=sig,new_cache=str(cache),input_hashes=hashes,
                deadline=deadline,weights=model['weights'],bank=model['bank'],cleanup='PRE_CLEANED',metrics=str(attempt/'metrics.json'))
            if proto:job.update(prototypes=proto['file'],prototype_sha256=ab.digest(proto['file']))
            env=ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT']=common['checkpoint']
            if stage=='eval':env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(model['weights'])/'adapter.pt'),
                OVERRIDE_INF_FEATURE_BANK_ROOT=model['bank'],OVERRIDE_INF_PROPOSALS_DIR=common['proposals'])
            log=logs/(key+'_'+attempt.name+'.log');queue.launch(job,attempt,env,log,cancel,deadline)
            artifacts=[attempt,log];result={}
            if stage=='eval':
                metric=ab.read(job['metrics']);old.validate_eval(metric,ab.read(model['metrics']),common['val_ids'],clean['count'],.2)
                old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
                report=ab.read(attempt/'verifier.json')
                if report['metrics']['confusion_matrix']!=metric['confusion_matrix']:raise ValueError('Independent metric mismatch')
                if method=='baseline':
                    old.mpsa.require_baseline_replay(ab.read(model['metrics']),metric)
                    old.require_image_replay(model['per_image'],str(attempt/'per_image_confusion.npz'))
                result=dict(report=str(attempt/'verifier.json'),per_image=str(attempt/'per_image_confusion.npz'))
            else:
                kind='samples' if stage=='sample' else 'evidence'
                artifacts += [cache/kind/stem for stem in ids]
            return result,artifacts
        return stages.run(key,action)
    def evaluate(item,gpu):
        method,seed=item;key=f'{method}_{seed}';result=execute(key,'eval',gpu,method=method,seed=seed)
        with guard:
            reports[key]=result;ab.write(output/'status.json',dict(status='running',completed=sorted(reports)))
    def fail(item,exc):raise exc
    def batch(stage,ids):
        queue.run_round([0,1],args.gpus,lambda shard,gpu:execute(f'{stage}_{shard}',stage,gpu,ids[shard::2]),cancel,deadline,fail)
    try:
        ab.write(output/'status.json',dict(status=status));ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,cache.parent,logs])),automatic_deletion=False))
        queue.run_round([('baseline',s) for s in old.SEEDS],args.gpus,evaluate,cancel,deadline,fail)
        execute('real_probe','sample',args.gpus[0],selected[:1])
        batch('sample',selected)
        queue.ensure_resources(deadline)
        proto=stages.run('prototypes',lambda attempt:build_prototypes(cache,selected,attempt))
        execute('map_probe','maps',args.gpus[0],common['val_ids'][:1])
        batch('maps',common['val_ids'])
        queue.run_round([(m,s) for s in old.SEEDS for m in ('bg1','bg8')],args.gpus,evaluate,cancel,deadline,fail)
        gmp.verify(records,'Historical inputs')
        if ab.source_identity()!=code:raise ValueError('Source changed')
        values={k:ab.read(v['report']) for k,v in reports.items()};summary={}
        for method in ('baseline','bg1','bg8'):
            scores=[values[f'{method}_{s}']['metrics']['mIoU'] for s in old.SEEDS]
            summary[method]=dict(mean=float(np.mean(scores)),sample_std=float(np.std(scores,ddof=1)),
                seeds={str(s):dict(metrics=values[f'{method}_{s}']['metrics'],
                    delta_pp=100*(scores[i]-values[f'baseline_{s}']['metrics']['mIoU']),
                    background_fp=sum(r['background_fp'] for r in values[f'{method}_{s}']['images']),
                    foreground_fn=sum(r['foreground_fn'] for r in values[f'{method}_{s}']['images'])) for i,s in enumerate(old.SEEDS)})
        for method in ('baseline','bg1','bg8'):
            for s in old.SEEDS:
                cell=summary[method]['seeds'][str(s)];baseline=summary['baseline']['seeds'][str(s)]
                cell['background_fp_delta']=cell['background_fp']-baseline['background_fp']
                cell['foreground_fn_delta']=cell['foreground_fn']-baseline['foreground_fn']
        ab.write(output/'summary.json',dict(status='complete',protocol=PROTOCOL,recipe=RECIPE,classes=values['baseline_42']['classes'],
            results=summary,reports=reports,evaluation_ids=common['val_ids'],note='Fixed VOC val development experiment. No statistical significance or automatic deployment.'))
        lines=['| Method | Mean mIoU % | Sample SD pp |','|---|---:|---:|']
        lines += [f'| {m} | {100*v["mean"]:.4f} | {100*v["sample_std"]:.4f} |' for m,v in summary.items()]
        (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status='complete'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except KeyboardInterrupt as exc:status='interrupted';error=str(exc);cancel.set();raise
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        ab.write(output/'status.json',dict(status=status,error=error,completed=sorted(reports)));queue.__file__=entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink();notify_experiment(args.version,status,str(output),time.time()-ab.read(budget)['started_at'])


if __name__=='__main__':main()
