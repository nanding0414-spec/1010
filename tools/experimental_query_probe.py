#!/usr/bin/env python3
"""Train-only foreground query representation probe; no MLP training or VOC val."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
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
PROTOCOL = 'query_foreground_probe_v1'


def normalize(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) < 1e-8).any():
        raise ValueError('Nonfinite/zero/malformed query representation')
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def choose_images(plan, split):
    classes = sorted({r['cls'] for rows in plan.values() for r in rows if r['cls'] != 'background'})
    if len(classes) != 20: raise ValueError('Expected 20 foreground classes')
    selected, coverage = {}, {}
    for partition, count, field in [('gallery',8,'train_images'),('heldout',4,'validation_images')]:
        chosen = set(); coverage[partition] = {}
        for cls in classes:
            candidates = [s for s in split[field] if any(r['cls']==cls for r in plan.get(s,[]))]
            candidates.sort(key=lambda s: hashlib.sha256(('42:'+s).encode()).hexdigest())
            if len(candidates) < (2 if partition=='gallery' else 1):
                raise ValueError(f'Insufficient fixed-split images: {partition}/{cls}')
            chosen.update(candidates[:count]); coverage[partition][cls]=len(candidates[:count])
        selected[partition]=sorted(chosen)
    if set(selected['gallery']) & set(selected['heldout']): raise ValueError('Image split leakage')
    return selected,coverage,classes


def filter_indices(proposer, masks, scores, labels, shape, postprocess, nms):
    """Mirror existing Active filtering while carrying exact original row indices."""
    valid=[]; cleaned={}
    for i,mask in enumerate(masks):
        mask=postprocess(mask,proposer.min_mask_region_area);area=int(mask.sum())
        if not area or area<proposer.min_mask_area: continue
        if proposer.max_mask_area_ratio and area>np.prod(shape)*proposer.max_mask_area_ratio: continue
        valid.append(i);cleaned[i]=mask
    kept=[]
    for cls in sorted(set(labels[i] for i in valid)):
        ids=[i for i in valid if labels[i]==cls]
        local=nms([cleaned[i] for i in ids],[scores[i] for i in ids],proposer.mask_nms_threshold) if proposer.mask_nms_threshold<1 else range(len(ids))
        kept.extend(ids[i] for i in local)
    kept.sort(key=lambda i:-scores[i])
    arrays=(np.stack([cleaned[i] for i in kept]) if kept else np.zeros((0,*shape),bool),
            np.asarray([scores[i] for i in kept],np.float32),np.asarray([labels[i] for i in kept],np.int32))
    return kept,arrays


class Capture:
    """Hook final decoder outputs, then verify every filter decision against Active."""
    def __init__(self,proposer):
        self.proposer=proposer;self.dimension=None
        ground,filter_fn=proposer._ground,proposer._filter
        def ground_hook(state,indices,expand):
            out=ground(state,indices,expand)
            if expand:
                import torch
                queries=out['queries']
                probs=out['pred_logits'].sigmoid().squeeze(-1)*out['presence_logit_dec'].sigmoid().reshape(len(indices),1)
                if queries.ndim!=3 or tuple(queries.shape[:2])!=tuple(probs.shape):
                    raise ValueError('Decoder query/mask slot shape mismatch')
                dim=int(queries.shape[-1])
                if self.dimension is not None and self.dimension!=dim: raise ValueError('Query dimension changed')
                self.dimension=dim
                for slot,cls in enumerate(indices):
                    selected=torch.where(probs[slot]>proposer.confidence_threshold)[0]
                    self.raw_queries.extend(queries[slot,selected].detach().float().cpu().numpy())
                    self.raw_scores.extend(probs[slot,selected].detach().float().cpu().tolist())
                    self.raw_labels.extend([cls+1]*len(selected))
                    self.raw_slots.extend([dict(class_id=cls+1,query_index=int(i),bucket=list(indices)) for i in selected.cpu().tolist()])
            return out
        def filter_hook(masks,scores,labels,shape):
            from src.mask_proposer_sam3 import _postprocess_small_regions,_nms_by_masks
            if labels!=self.raw_labels or scores!=self.raw_scores or len(masks)!=len(self.raw_queries):
                raise ValueError('Pre-filter mask/query order mismatch')
            result=filter_fn(masks,scores,labels,shape)
            ids,replica=filter_indices(proposer,masks,scores,labels,shape,_postprocess_small_regions,_nms_by_masks)
            if any(not np.array_equal(a,b) for a,b in zip(result,replica)):
                raise ValueError('Indexed Active filtering does not reproduce original')
            if proposer.max_masks: ids=ids[:proposer.max_masks]
            self.queries=(np.stack([self.raw_queries[i] for i in ids]).astype(np.float32)
                          if ids else np.empty((0,self.dimension or 0),np.float32))
            self.slots=[self.raw_slots[i] for i in ids]
            return result
        proposer._ground,proposer._filter=ground_hook,filter_hook

    def __call__(self,rgb,classes):
        self.raw_queries=[];self.raw_scores=[];self.raw_labels=[];self.raw_slots=[]
        result=self.proposer(np.ascontiguousarray(rgb[...,::-1]),class_indices=classes)['instances']
        masks=result.pred_masks.cpu().numpy().astype(bool)
        scores=result.scores.float().cpu().numpy()
        labels=self.proposer.last_result['class_ids']
        if len(masks)!=len(self.queries): raise ValueError('Final query/mask count mismatch')
        if len(masks): normalize(self.queries)
        return masks,scores,labels,self.queries.copy(),self.slots.copy()


def regional_query(region,cls,masks,scores,labels,queries):
    """Assign each region pixel to its highest-score same-class candidate."""
    owner=np.full(region.shape,-1,np.int32);best=np.full(region.shape,-np.inf,np.float32)
    for i in np.flatnonzero(labels==cls):
        update=region & masks[i] & (scores[i]>best)
        owner[update]=i;best[update]=scores[i]
    if (owner[region]<0).any(): raise ValueError('Foreground region not covered by same-class replay')
    ids,counts=np.unique(owner[region],return_counts=True)
    vector=(normalize(queries[ids])*(counts/counts.sum())[:,None]).sum(0)
    return normalize(vector[None])[0],dict(candidates=ids.tolist(),pixel_counts=counts.tolist())


def match_full(region,masks):
    """Class-blind geometry matching. GT/pseudo category never selects a query."""
    if not len(masks): return -1,0.
    inter=np.logical_and(masks,region).sum((1,2));union=np.logical_or(masks,region).sum((1,2))
    iou=inter/np.maximum(union,1);index=int(np.argmax(iou))
    return (index if iou[index]>=.5 else -1),float(iou[index])


def write_npz(path,**arrays):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(temporary,path)


def worker(job):
    sys.path.insert(0,str(ROOT/'sam3-main'))
    import torch
    from PIL import Image
    from src.region_extractors import get_regions_from_pseudo
    started=time.monotonic();status='failed';category='global';error=None
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    try:
        ab.install_precision()
        from project_utils.seed_utils import set_global_seed
        set_global_seed(42)
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
        r=job['recipe']
        generator=SAM3PseudoMaskGenerator(dataset_type='voc',ckpt=job['checkpoint'],device='cuda:0',
            conf_thr=r['conf_thr'],min_mask_area=r['min_mask_area'],use_bf16=False,
            active_options=r['active_options'],dataset_root=job['dataset_root'],image_dir=job['image_dir'],
            imagelevel_json=job['imagelevel_json'],output_dir=job['cache_root'])
        if old.canonical([asdict(c) for c in generator.active_classes])!=r['active_classes'] or generator.classes!=r['prompts']:
            raise ValueError('Historical vocabulary changed')
        _,labels=generator.get_image_list_and_labels(r['split'])
        capture=Capture(generator.active_proposer)
        stages=ab.Stages(Path(job['cache_root']),True)
        for stem in job['ids']:
            queue.ensure_resources(job['deadline'])
            def image_action(attempt):
                image_path=Path(job['images'][stem]);pseudo=Path(job['pseudo'])/(stem+'.png')
                ident=job['cache_identity'][stem]
                if ab.digest(image_path)!=ident['image_sha256'] or ab.digest(pseudo)!=ident['mask_source_sha256']:
                    raise ValueError('Training input changed')
                with Image.open(image_path) as im: rgb=np.asarray(im.convert('RGB'))
                with Image.open(pseudo) as im: seg=np.asarray(im)
                image_labels=labels.get(str(image_path),labels.get(stem))
                if image_labels is None: raise ValueError('Missing training image-level label')
                allowed=[int(c) for c in dict.fromkeys(image_labels) if int(c)!=0]
                masks,scores,cls,q,slots=capture(rgb,[c-1 for c in allowed])
                regions=get_regions_from_pseudo(image_path,300,job['pseudo'])
                old.region_scores(seg,masks,scores,cls,allowed,regions)  # strict original PNG equality
                rows=job['plan'][stem];vectors=[];bindings=[]
                for row in rows:
                    reg=regions[row['index']]
                    if old.ab_hash_mask(reg['mask'])!=row['mask_sha256']: raise ValueError('Region identity changed')
                    vec,binding=regional_query(reg['mask'].astype(bool),reg['cls_id'],masks,scores,cls,q)
                    vectors.append(vec);bindings.append(binding)
                conditional=np.stack(vectors)
                is_heldout=stem in job['heldout']
                full=np.zeros_like(conditional);valid=np.zeros(len(rows),bool);ious=np.zeros(len(rows),np.float32)
                prompt_classes=np.full(len(rows),-1,np.int32);matches=[]
                if is_heldout:
                    fm,fs,fc,fq,fslots=capture(rgb,None)  # all VOC words; no image-level class restriction
                    for i,row in enumerate(rows):
                        index,iou=match_full(regions[row['index']]['mask'].astype(bool),fm)
                        ious[i]=iou
                        if index>=0:
                            full[i]=normalize(fq[index:index+1])[0];valid[i]=True;prompt_classes[i]=int(fc[index])-1
                        matches.append(dict(index=index,iou=iou,slot=None if index<0 else fslots[index]))
                cache=queue.cache_read(Path(job['cache'])/'train'/stem,ident)
                if cache is None: raise ValueError('Missing original GMP cache')
                with np.load(cache['file'],allow_pickle=False) as data:
                    lookup={(str(c),str(s)):v for c,s,v in zip(data['classes'],data['ids'],data['C'])}
                    gmp_vectors=np.stack([lookup[(row['cls'],row['id'])] for row in rows])
                path=attempt/'vectors.npz'
                write_npz(path,conditional=conditional,full=full,valid=valid,match_iou=ious,
                    prompt_classes=prompt_classes,gmp=normalize(gmp_vectors),
                    classes=np.asarray([row['cls'] for row in rows]),ids=np.asarray([row['id'] for row in rows]))
                diag=attempt/'mapping.json'
                ab.write(diag,dict(image=stem,png_equal=True,query_dim=capture.dimension,
                    original_proposal_hashes=queue.mask_hashes(masks),original_slots=slots,
                    regions=rows,conditional_binding=bindings,full_vocabulary_matches=matches,
                    note='Heldout geometry matches pseudo regions; diagnostic, not deployed inference.'))
                return dict(file=str(path),mapping=str(diag)),[path,diag]
            stages.run(stem,image_action)
            print('[Query probe]',stem,flush=True)
        status='complete'
    except queue.Deferred as exc: category='deferred';error=str(exc);raise
    except torch.cuda.OutOfMemoryError as exc: category='oom';error=str(exc);raise
    except BaseException as exc: error=str(exc);raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,category=category,error=error,
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),precision_policy=ab.POLICY))


def predict(gallery,labels,queries):
    """Fixed top-5 cosine vote, ties: summed similarity then alphabetical class."""
    gallery,queries=normalize(gallery),normalize(queries)
    result=[]
    for start in range(0,len(queries),128):
        for sims in queries[start:start+128]@gallery.T:
            ids=np.argsort(-sims,kind='stable')[:min(5,len(sims))]
            options=sorted(set(labels[ids]))
            result.append(max(options,key=lambda c:(int((labels[ids]==c).sum()),float(sims[ids][labels[ids]==c].sum()),-options.index(c))))
    return np.asarray(result)


def summarize(root,selected,classes,prompts):
    def load(ids):
        values={}
        for stem in ids:
            record=ab.read(root/stem/'complete.json');gmp.verify(record['artifacts'],'Probe image')
            with np.load(record['result']['file'],allow_pickle=False) as x:
                for k in x.files: values.setdefault(k,[]).append(x[k].copy())
        return {k:np.concatenate(v) for k,v in values.items()}
    gallery,held=load(selected['gallery']),load(selected['heldout'])
    valid=held['valid'];truth=held['classes'];outputs={};results={}
    def metrics(pred,mask):
        by_class={c:dict(count=int((truth[mask]==c).sum()),accuracy=float(np.mean(pred[truth[mask]==c]==c))
                        if (truth[mask]==c).any() else None) for c in classes}
        present=[v['accuracy'] for v in by_class.values() if v['accuracy'] is not None]
        return dict(count=int(mask.sum()),accuracy=float(np.mean(pred==truth[mask])) if mask.any() else None,
            macro_accuracy_present_classes=float(np.mean(present)) if present else None,
            evaluated_class_count=len(present),by_class=by_class)
    for mode in ('conditional','full'):
        mask=np.ones(len(truth),bool) if mode=='conditional' else valid
        for method in ('gmp','query','concat'):
            gx=gallery['gmp'] if method=='gmp' else gallery['conditional']
            hx=held['gmp'][mask] if method=='gmp' else held[mode][mask]
            if method=='concat':
                gx=np.concatenate([normalize(gallery['gmp']),normalize(gallery['conditional'])],axis=1)
                hx=np.concatenate([normalize(held['gmp'][mask]),normalize(held[mode][mask])],axis=1) if mask.any() else np.empty((0,gx.shape[1]))
            pred=predict(gx,gallery['classes'],hx) if mask.any() else np.asarray([],dtype=str)
            outputs[mode+'_'+method]=pred
            results[mode+'_'+method]=metrics(pred,mask)
    direct=np.asarray([prompts[int(i)] for i in held['prompt_classes'][valid]])
    results['full_prompt_class']=metrics(direct,valid)
    a=outputs['full_gmp']==truth[valid];b=outputs['full_query']==truth[valid]
    cosine=(normalize(held['conditional'][valid])*normalize(held['full'][valid])).sum(1) if valid.any() else np.array([])
    return dict(status='complete',protocol=PROTOCOL,selected=selected,query_dimension=gallery['conditional'].shape[1],
        gallery_regions=len(gallery['classes']),heldout_regions=len(truth),matched_full_regions=int(valid.sum()),
        full_match_coverage=float(valid.mean()),coverage_by_class={c:float(valid[truth==c].mean()) for c in classes},
        retrieval=results,full_error_overlap=dict(both_correct=int((a&b).sum()),gmp_only=int((a&~b).sum()),
            query_only=int((~a&b).sum()),both_wrong=int((~a&~b).sum())),
        prompt_shift_cosine_quantiles=np.quantile(cosine,[0,.25,.5,.75,1]).tolist() if len(cosine) else None,
        missing_count=int((~valid).sum()),
        note='Foreground-only pseudo-label retrieval. Conditional queries use known classes; full-vocabulary matching uses pseudo geometry. Cosine shift also includes candidate geometry/context changes, not isolated prompt sensitivity. No GT supervision, background classifier, MLP training or VOC val mIoU. No automatic full experiment.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--previous',type=Path,default=Path('experiments/gmp_mlp2_20261002_141439'))
    parser.add_argument('--version',default='query_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    parser.add_argument('--max-hours',type=float,default=12)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or not math.isfinite(args.max_hours) or args.max_hours<=0:parser.error('Invalid GPUs/hours')
    for gpu in args.gpus:ab.check_gpu(gpu)
    common,refs,split,records,clean,raw,sig=old.checked_history(args.previous.resolve(),Path(old.mpsa.DEFAULT_CLEAN))
    recipe,meta=old.generation_recipe(common);records+=meta
    source=Path(ab.read(args.previous/'manifest.json')['combo_previous'])
    queue.verified_stage(source,'membership',records)
    plan=ab.read(source/'members.json')
    selected,coverage,classes=choose_images(plan,split)
    if set(classes)!=set(recipe['prompts']):raise ValueError('Feature-bank class names and prompt IDs differ')
    ids=selected['gallery']+selected['heldout'];plan={s:[r for r in plan[s] if r['cls']!='background'] for s in ids}
    for stem in ids:
        cache=queue.cache_read(Path(common['cache'])/'train'/stem,common['cache_identity'][stem])
        if cache is None:raise ValueError('Missing train feature cache')
        records+=cache['artifacts']+ab.identities([Path(common['cache'])/'train'/stem/'complete.json'])
    code=ab.source_identity();identity=old.canonical(dict(protocol=PROTOCOL,previous=str(args.previous.resolve()),
        selected=selected,coverage=coverage,plan=plan,recipe=recipe,code=code,references=records,
        matching='class_blind_best_iou_ge_0.5',query_binding='highest_score_pixel_owner_same_class',gpus=args.gpus))
    output=ROOT/'experiments'/args.version;cache_root=ROOT/'storage/raw_banks/voc'/args.version/'query_probe'
    logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Changed version or --resume missing')
    elif cache_root.parent.exists() or logs.exists():raise FileExistsError('Version already used')
    stages=ab.Stages(output,True);stages.verify_all();ab.Stages(cache_root,True).verify_all()
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+2**30:raise queue.Deferred('Need 1 GiB probe allowance plus 5 GiB reserve')
    print('[Plan] train-only foreground query probe:',{k:len(v) for k,v in selected.items()},flush=True)
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
    previous_entry=queue.__file__;queue.__file__=__file__;status='running';error=None
    def execute(name,gpu,batch):
        def action(attempt):
            job=dict(common,stage='query_probe',attempt=str(attempt),recipe=recipe,plan=plan,
                ids=batch,heldout=selected['heldout'],cache_root=str(cache_root),deadline=deadline)
            env=ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT']=common['checkpoint']
            log=logs/(name+'_'+attempt.name+'.log')
            queue.launch(job,attempt,env,log,cancel,deadline)
            return {},[attempt,log]
        return stages.run(name,action)
    try:
        ab.write(output/'status.json',dict(status=status))
        ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,cache_root.parent,logs])),automatic_deletion=False))
        probe=selected['gallery'][:1]+selected['heldout'][:1]
        execute('probe',args.gpus[0],probe)
        remaining=[i for i in ids if i not in probe]
        queue.run_round([s for s in (0,1) if not (output/f'shard_{s}'/'complete.json').exists()],
            args.gpus,lambda shard,gpu:execute(f'shard_{shard}',gpu,remaining[shard::2]),
            cancel,deadline,lambda item,exc:(_ for _ in ()).throw(queue.GlobalFailure(str(exc))))
        result=summarize(cache_root,selected,classes,recipe['prompts'])
        gmp.verify(records,'Historical inputs');ab.Stages(cache_root,True).verify_all()
        if ab.source_identity()!=code:raise ValueError('Code changed during probe')
        result['workers']={p.parent.parent.name+'/'+p.parent.name:ab.read(p) for p in output.glob('*/attempt_*/queue_worker.json')}
        ab.write(output/'summary.json',result)
        lines=['| Representation | Regions | Accuracy % | Macro accuracy % | Classes |','|---|---:|---:|---:|---:|']
        fmt=lambda x:'—' if x is None else f'{100*x:.3f}'
        for name,v in result['retrieval'].items():
            lines.append(f'| {name} | {v["count"]} | {fmt(v["accuracy"])} | {fmt(v["macro_accuracy_present_classes"])} | {v["evaluated_class_count"]} |')
        (output/'summary.md').write_text('Foreground query probe; NOT VOC val mIoU.\n\n'
            +f'Full-vocabulary geometric match coverage: {100*result["full_match_coverage"]:.2f}%.\n\n'
            +'\n'.join(lines)+'\n\n'+result['note']+'\n',encoding='utf-8')
        status='complete'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        ab.write(output/'status.json',dict(status=status,error=error))
        queue.__file__=previous_entry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink()
    print(f'[{status}] {output}',flush=True)


if __name__=='__main__':main()
