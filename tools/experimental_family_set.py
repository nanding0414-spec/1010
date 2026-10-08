"""View-balanced family-set retrieval with frozen real centroids; no training."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_view_augmentation_queue as views
ab,queue,gmp,old=views.ab,views.queue,views.gmp,views.old
PROTOCOL='view_balanced_family_set_v1'
METHODS=('real_flat','view30_flat_reference','view30_flat_fixed','real_set','view30_family_set','view30_random_set')
RULE=dict(temperature=.05,group='source_family_class',background='unchanged singletons',
          pooling='equal view mass; mean exp within view; stabilized logmeanexp',
          candidates='existing top210 vector shortlist; score full members of every shortlisted set',
          class_score='top5 distinct candidate sets per class; .6 local + .4 original real centroid',
          random_seed=42,random_control='permute members within class/view; preserve every set size and view composition',
          hcr=False,training=False,support_filter=False,angles=['original','L30','R30'],gt='post-hoc only')


def view_logmeanexp(similarities, view_ids, temperature=.05):
    x=np.asarray(similarities,np.float64);v=np.asarray(view_ids)
    if x.ndim!=1 or not len(x) or v.shape!=x.shape or not np.isfinite(x).all():raise ValueError('Invalid set similarities/views')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    if not np.issubdtype(v.dtype,np.integer) or (v<0).any():raise ValueError('View IDs must be nonnegative integers')
    if len(x)==1:return float(x[0])
    maximum=float(x.max());exp=np.exp((x-maximum)/temperature)
    mass=np.mean([exp[v==i].mean() for i in np.unique(v)])
    return maximum+temperature*float(np.log(mass))


def random_groups(groups,labels,view_ids,seed=42):
    groups=np.asarray(groups,np.int64);labels=np.asarray(labels);v=np.asarray(view_ids)
    if groups.ndim!=1 or groups.shape!=labels.shape or v.shape!=groups.shape:raise ValueError('Invalid random-group shapes')
    result=groups.copy()
    for c in np.unique(labels):
        if c=='background':continue
        for view in np.unique(v[labels==c]):
            rows=np.flatnonzero((labels==c)&(v==view))
            key=f'{c}|{int(view)}|{seed}'.encode();salt=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')
            result[rows]=groups[np.random.default_rng(salt).permutation(rows)]
    if Counter(zip(result,labels,v))!=Counter(zip(groups,labels,v)):raise ValueError('Random group composition changed')
    return result


def member_layout(job):
    from src.feature_bank_storage import load_feature_matrix_npz
    from src.adapter_training import image_id_from_sample
    mapping=ab.read(Path(job['real_bank'])/'meta/class_mapping.json')['id_to_dirname']
    order=[mapping[k] for k in sorted(mapping,key=int) if int(k)!=0]+[mapping['0']]
    plan=ab.read(job['selection'])
    if ab.digest(job['selection'])!=job['selection_sha256']:raise ValueError('Selection changed')
    expanded=job['method']!='real_set'
    gallery=[];real_gallery=[];labels=[];real_labels=[];names=[];vid=[];group_keys={}
    for c in order:
        real_files=list((Path(job['real_bank'])/'features'/c).glob('*.npz'))
        files=list((Path(job['bank'])/'features'/c).glob('*.npz'))
        if len(real_files)!=1 or len(files)!=1:raise ValueError('Class bank missing/ambiguous')
        original,ids=load_feature_matrix_npz(real_files[0]);x,new_ids=load_feature_matrix_npz(files[0])
        if not np.array_equal(x[:len(original)],original) or not np.array_equal(new_ids[:len(ids)],ids):raise ValueError('Original rows changed')
        rows=[(image_id_from_sample(i),0) for i in ids]
        if expanded:
            additions=[(i,r) for i,r in enumerate(plan['rows']['view30']) if r['cls']==c]
            expected=[f"{r['family']}_mv_view30_{i:06d}" for i,r in additions]
            if new_ids[len(ids):].tolist()!=expected:raise ValueError('Synthetic row order/IDs changed')
            rows += [(r['family'],1 if r['view']=='L30' else 2) for _,r in additions]
        elif len(x)!=len(original):raise ValueError('Unexpected original set members')
        for index,(family,view) in enumerate(rows):
            key=(c,family) if c!='background' else (c,str(new_ids[index]))
            if key not in group_keys:group_keys[key]=len(group_keys)
            names.append(group_keys[key]);vid.append(view)
        if len(rows)!=len(x):raise ValueError('Gallery/member alignment failed')
        gallery.append(x);labels += [c]*len(x)
        real_gallery.append(original);real_labels += [c]*len(original)
    features=np.concatenate(gallery).astype(np.float32)
    features/=np.linalg.norm(features,axis=1,keepdims=True)+1e-8
    original=np.concatenate(real_gallery).astype(np.float32)
    original/=np.linalg.norm(original,axis=1,keepdims=True)+1e-8
    groups=np.asarray(names,np.int64);vid=np.asarray(vid,np.int64);labels=np.asarray(labels)
    before=groups.copy()
    if job['method']=='view30_random_set':groups=random_groups(groups,labels,vid)
    buckets=defaultdict(list)
    for row,group in enumerate(groups):buckets[int(group)].append(row)
    members={group:np.asarray(rows,np.int64) for group,rows in buckets.items()}
    # Verify that shuffling never mixes class labels in one group.
    if any(len(np.unique(labels[r]))!=1 for r in members.values()):raise ValueError('Mixed class set')
    diagnostic=dict(vector_count=len(features),set_count=len(members),view_counts=dict(Counter(map(int,vid))),
        set_size_quantiles=np.quantile([len(r) for r in members.values()],[0,.5,.9,1]).tolist(),
        regrouped_fraction=float(np.mean(before!=groups)),rule=RULE)
    return features,labels,groups,vid,members,original,real_labels,diagnostic


def score_sets(query,gallery,shortlist,groups,view_ids,members,labels,context,class_topm=5,centroid_weight=.4):
    from src.retrieval import RetrievalPrediction
    if class_topm<1 or not 0<=centroid_weight<=1:raise ValueError('Invalid class scoring')
    query=np.asarray(query,np.float32);centers=query@context.centroids.T
    predictions=[];diagnostics=[]
    for k,(q,indices) in enumerate(zip(query,shortlist)):
        indices=np.asarray(indices);indices=indices[indices>=0]
        if not len(indices) or indices.max()>=len(gallery):raise ValueError('Invalid candidate shortlist')
        candidates=np.unique(groups[indices]);scores=defaultdict(list)
        for group in candidates:
            rows=members[int(group)];similarity=gallery[rows]@q
            value=view_logmeanexp(similarity,view_ids[rows],RULE['temperature'])
            scores[str(labels[rows[0]])].append(value)
        local=np.asarray([np.mean(sorted(scores[c],reverse=True)[:class_topm]) if scores[c] else float(centers[k,i])
                          for i,c in enumerate(context.labels)],np.float32)
        fused=(1-centroid_weight)*local+centroid_weight*centers[k]
        ranked=np.argsort(fused)[::-1];a,b=ranked[:2]
        predictions.append(RetrievalPrediction(context.labels[a],float(fused[a]),float(fused[a]-fused[b])))
        diagnostics.append(dict(candidate_sets=len(candidates),max_family_slots=max(Counter(groups[indices]).values()),
            winner_local=float(local[a]),winner_center=float(centers[k,a])))
    return predictions,diagnostics


def install_retrieval(job):
    base=queue.install_cached_eval
    def install(j):
        base(j);queue.KEY_IMAGES=[]
        import run_inference as inference
        import src.inference as inf
        import src.retrieval as ret
        from src.feature_bank_storage import load_feature_matrix_npz
        state={};trace=[];original_classify=inf.classify_queries
        method=job['method'];layout=None;frozen=None
        if method not in ('real_flat','view30_flat_reference'):
            # Family layout provides exact row identities for both flat and set controls.
            if method=='view30_flat_fixed':job_for_layout=dict(job,method='view30_family_set')
            else:job_for_layout=job
            layout=member_layout(job_for_layout)
            frozen=ret.build_retrieval_context(layout[5],layout[6])
            ab.write(Path(job['attempt'])/'group_diagnostics.json',layout[7])
        load=inference.load_strict_proposals
        def read(*args,**kwargs):
            masks,scores=load(*args,**kwargs);state.clear();state.update(id=args[1],masks=masks)
            return masks,scores
        inference.load_strict_proposals=read
        build=inference.build_feature_extractor
        def extractor(*args,**kwargs):
            model=build(*args,**kwargs);extract=model.extract_features_batch
            def batch(rgb,masks):
                values=extract(rgb,masks);state['valid']=[i for i,v in enumerate(values) if np.isfinite(v).all() and np.linalg.norm(v)>=1e-6]
                return values
            model.extract_features_batch=batch;return model
        inference.build_feature_extractor=extractor
        verified=False
        def classify(*args,**kwargs):
            nonlocal verified
            if args:raise ValueError('Expected keyword retrieval inputs')
            if kwargs['mode']!='balanced' or kwargs['class_topm']!=5 or kwargs['centroid_weight']!=.4:raise ValueError('Retrieval parameters changed')
            diagnostic=None
            if layout is not None:
                if not verified:
                    if not np.array_equal(kwargs['all_feats'],layout[0]) or list(kwargs['all_labels'])!=layout[1].tolist():raise ValueError('Live gallery/member order differs')
                    if tuple(kwargs['context'].labels)!=frozen.labels:raise ValueError('Class center order differs')
                    verified=True
                kwargs=dict(kwargs,context=frozen)
            if method.endswith('_set'):
                count=min(max(kwargs['topk'],kwargs['class_topm']*len(frozen.labels)*kwargs['search_multiplier']),len(layout[0]))
                _,indices=ret._search(kwargs['query_feats'],kwargs['all_feats'],kwargs['faiss_index'],count)
                result,diagnostic=score_sets(kwargs['query_feats'],layout[0],indices,layout[2],layout[3],layout[4],layout[1],frozen)
            else:result=original_classify(**kwargs)
            if len(result)!=len(state['valid']):raise ValueError('Query-mask trace alignment differs')
            for n,(r,index) in enumerate(zip(result,state['valid'])):
                trace.append(dict(image=state['id'],proposal_index=index,mask_sha256=queue.mask_hashes([state['masks'][index]])[0],
                    label=r.label,score=r.score,margin=r.margin,**(diagnostic[n] if diagnostic else {})))
            return result
        inf.classify_queries=classify;job['_trace']=trace
    queue.install_cached_eval=install


def posthoc(job):
    from PIL import Image
    trace=job['_trace'];by_image=defaultdict(list)
    for row in trace:by_image[row['image']].append(row)
    counts=Counter();margins=[]
    for stem,rows in by_image.items():
        with np.load(Path(job['proposals'])/(stem+'.npz'),allow_pickle=False) as f:masks=f['masks']
        with Image.open(Path(job['gt_dir'])/(stem+'.png')) as image:gt=np.asarray(image)
        for row in rows:
            mask=masks[row['proposal_index']].astype(bool)
            if mask.shape!=gt.shape or queue.mask_hashes([mask])[0]!=row['mask_sha256']:raise ValueError('Posthoc geometry/identity differs')
            pixels=gt[mask&(gt!=255)];counts['candidates']+=1
            if not len(pixels):counts['empty_gt']+=1;continue
            histogram=np.bincount(pixels,minlength=21);label=int(histogram.argmax());purity=float(histogram[label]/len(pixels))
            if purity<.9:counts['mixed']+=1;continue
            counts['pure']+=1;margins.append(row['margin'])
            correct=row['label']==views.provenance.evidence.VOC_CLASSES[label]
            counts['pure_correct']+=int(correct)
            counts['pure_background_fp']+=int(label==0 and row['label']!='background')
            counts['pure_foreground_fn']+=int(label!=0 and row['label']=='background')
            counts['pure_foreground_confusion']+=int(label!=0 and row['label']!='background' and not correct)
    ab.write(Path(job['attempt'])/'retrieval_trace.json',trace)
    ab.write(Path(job['attempt'])/'candidate_diagnostics.json',dict(counts=dict(counts),
        pure_accuracy=counts['pure_correct']/max(1,counts['pure']),mean_pure_margin=float(np.mean(margins)) if margins else None,
        note='GT read only after all predictions/metrics were produced; purity>=.9 is diagnostic only'))


def worker(job):
    import torch
    start=time.monotonic();status='failed';error=None;category='local'
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    try:
        if job['protocol']!=PROTOCOL or job['method'] not in METHODS or job['stage']!='eval' or job['hcr']:raise ValueError('Invalid set experiment job')
        install_retrieval(job);old.previous.worker(job);posthoc(job)
        torch.cuda.synchronize();status='complete'
    except queue.Deferred as exc:category='deferred';error=str(exc);raise
    except torch.cuda.OutOfMemoryError as exc:category='oom';error=str(exc);raise
    except (ValueError,KeyError,FileNotFoundError,queue.GlobalFailure) as exc:category='global';error=str(exc);raise
    except BaseException as exc:error=str(exc);raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,error=error,category=category,
            seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def summarize(results):
    groups={};contrasts={}
    for method in METHODS:
        scores={str(s):results[f'{method}_{s}']['miou'] for s in old.SEEDS if f'{method}_{s}' in results}
        groups[method]=dict(complete=len(scores)==3,seeds=scores,mean=float(np.mean(list(scores.values()))) if len(scores)==3 else None,
            sample_std=float(np.std(list(scores.values()),ddof=1)) if len(scores)==3 else None)
    for a,b in [('real_set','real_flat'),('view30_flat_fixed','view30_flat_reference'),('view30_family_set','view30_flat_fixed'),
                ('view30_family_set','real_set'),('view30_family_set','view30_random_set')]:
        if groups[a]['complete'] and groups[b]['complete']:
            delta={s:100*(groups[a]['seeds'][s]-groups[b]['seeds'][s]) for s in groups[a]['seeds']}
            contrasts[a+'-'+b]=dict(by_seed_pp=delta,mean_pp=float(np.mean(list(delta.values()))))
    return dict(groups=groups,contrasts=contrasts,results=results,note='Fixed VOC val development batch; no significance or deployment claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--version',default='family_set_20261008')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=24)
    args=p.parse_args();os.chdir(ROOT);views.install_relocation()
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if not np.isfinite(args.max_hours) or args.max_hours<=0:raise ValueError('Invalid time budget')
    previous=ROOT/'experiments/multiview_bank_20261008';summary=ab.read(previous/'summary.json');manifest=ab.read(previous/'manifest.json')
    if summary['status']!='complete' or len(summary['results'])!=24:raise ValueError('Complete matched-gallery history required')
    gmp.verify(views.unique_records(manifest['code']),'Frozen previous code')
    selection=previous/'selection.json';records=ab.identities([previous/'manifest.json',previous/'summary.json',selection]);jobs={};envs={};refs={}
    selection_marker=previous/'selection/complete.json';gmp.verify(ab.read(selection_marker)['artifacts'],'Matched view selection')
    records+=ab.identities([selection_marker])
    print('[Preflight] checking model/gallery artifacts and historical replays',flush=True)
    for seed in old.SEEDS:
        real=summary['results'][f'real_clean_{seed}'];expanded=summary['results'][f'view30_clean_{seed}']
        attempt=Path(real['metrics']).parent;job=ab.read(attempt/'job.json');env=ab.read(attempt/'command.json')['environment']
        if job['hcr'] or job['role']!='gmp2':raise ValueError('Wrong historical model')
        if ab.read(Path(real['bank'])/'meta/feature_signature.json')!=ab.read(Path(expanded['bank'])/'meta/feature_signature.json'):raise ValueError('Different embedding spaces')
        for key in (f'build_{seed}',f'real_clean_{seed}',f'view30_clean_{seed}'):
            marker=previous/key/'complete.json';value=ab.read(marker)
            gmp.verify(views.unique_records(value['artifacts']),'Completed bank/evaluation');records+=value['artifacts']+ab.identities([marker])
        job.update(real_bank=real['bank'],expanded_bank=expanded['bank'],selection=str(selection),selection_sha256=ab.digest(selection))
        jobs[seed]=job;envs[seed]=env;refs[seed]=dict(real_flat=real,view30_flat_reference=expanded)
    records=views.unique_records(records);code=ab.source_identity()
    identity=old.canonical(dict(protocol=PROTOCOL,rule=RULE,code=code,records=records,max_hours=args.max_hours))
    output=ROOT/'experiments'/args.version;logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Existing version differs/needs resume')
    else:
        if logs.exists():raise FileExistsError('Log version already used')
        ab.write(output/'manifest.json',identity)
    for gpu in (0,1):ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    stages=views.AliasAwareStages(output,True);cancel=threading.Event();guard=threading.Lock()
    start=time.time();deadline=start+args.max_hours*3600;status='running';error=None;results={};failures={}
    queue.__file__=__file__
    def stop(*_):cancel.set();raise KeyboardInterrupt('Stopped')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),failures=failures,expected=18))
        ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,**summarize(results)))
    def evaluate(item,gpu):
        method,seed=item;key=f'{method}_{seed}'
        def action(attempt):
            job=copy.deepcopy(jobs[seed]);job.update(protocol=PROTOCOL,method=method,stage='eval',hcr=False,
                attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline)
            job['bank']=job['real_bank'] if method in ('real_flat','real_set') else job['expanded_bank']
            env=os.environ.copy();env.update(envs[seed]);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env=views.evaluation_environment(job,env);env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=logs/f'{key}_{attempt.name}.log';queue.launch(job,attempt,env,log,cancel,deadline)
            return dict(attempt=str(attempt)),[attempt,log]
        result=stages.run(key,action);attempt=Path(result['attempt']);metric=ab.read(attempt/'metrics.json')
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
        with np.load(attempt/'per_image_confusion.npz',allow_pickle=False) as f:
            if len(f['ids'])!=1449 or f['ids'].tolist()!=jobs[seed]['val_ids']:raise ValueError('Validation IDs differ')
        if 'semantic_rescue' in metric:raise ValueError('HCR unexpectedly enabled')
        if method in refs[seed]:
            reference=refs[seed][method]
            old.require_image_replay(reference['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=ab.read(reference['metrics'])['confusion_matrix']:raise ValueError('Historical replay differs')
        corrected=views.provenance.evidence.checked_copy(metric)[0]
        with guard:
            results[key]=dict(miou=corrected['mIoU'],per_class_iou=corrected['per_class_iou'],metrics=str(attempt/'metrics.json'),
                per_image=str(attempt/'per_image_confusion.npz'),candidates=ab.read(attempt/'candidate_diagnostics.json'));flush()
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        queue.run_round([(m,s) for s in old.SEEDS for m in METHODS[:2]],(0,1),evaluate,cancel,deadline,fatal)
        def failed(item,exc):
            with guard:failures[str(item)]=str(exc);flush()
        queue.run_round([(m,s) for s in old.SEEDS for m in METHODS[2:]],(0,1),evaluate,cancel,deadline,failed)
        gmp.verify(code,'Frozen source');stages.verify_all();status='complete' if len(results)==18 else 'partial'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();lock.unlink();ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(logs)],automatic_deletion=False))
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
