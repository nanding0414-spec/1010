"""Fixed train-only gallery trust / foreground rejection factorial experiment."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_bank_provenance as provenance
ab,queue,gmp,old=provenance.ab,provenance.queue,provenance.gmp,provenance.old
evidence=provenance.evidence
PREVIOUS=ROOT/'experiments/bank_provenance_20261005_101808'
RECIPES={'baseline':(False,None),'trust':(True,None),'reject01':(False,.01),
         'reject05':(False,.05),'trust_reject01':(True,.01),'trust_reject05':(True,.05)}
TAU=.05
PENALTY=.05


def reliability(scores, own):
    scores=np.asarray(scores,np.float32);own=np.asarray(own,int)
    if scores.ndim!=2 or len(scores)!=len(own) or scores.shape[1]<2 or not np.isfinite(scores).all():
        raise ValueError('Invalid class supports')
    if (own<0).any() or (own>=scores.shape[1]).any():raise ValueError('Invalid labels')
    other=scores.copy();other[np.arange(len(own)),own]=-np.inf
    delta=scores[np.arange(len(own)),own]-other.max(1)
    return (1/(1+np.exp(-np.clip(delta/TAU,-60,60)))).astype(np.float32)


def class_thresholds(scores,own,names,quantile):
    if quantile not in (.01,.05):raise ValueError('Unregistered quantile')
    result=[]
    for c in range(len(names)):
        values=np.asarray(scores)[np.asarray(own)==c,c]
        if not len(values) or not np.isfinite(values).all():raise ValueError('Missing calibration class')
        result.append(float(np.quantile(values,quantile)))
    return result


def supports(features,labels,images,trust=None,device='cuda:0',deadline=float('inf')):
    """Full class top-five mean; exclude every region from the query's image.

    Trust penalizes the original cosine top-five, rather than changing membership.
    Only this training-only function reads gallery metadata, never segmentation GT.
    """
    import torch
    x=np.asarray(features,np.float32);labels=np.asarray(labels);images=np.asarray(images)
    if x.ndim!=2 or len(x)!=len(labels) or len(x)!=len(images) or not np.isfinite(x).all():raise ValueError('Invalid supports input')
    if not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4):raise ValueError('Unit features required')
    if trust is not None and (np.shape(trust)!=(len(x),) or not np.isfinite(trust).all() or (trust<0).any() or (trust>1).any()):raise ValueError('Invalid trust')
    names=list(dict.fromkeys(labels.tolist()));own=np.array([names.index(c) for c in labels])
    _,img=np.unique(images,return_inverse=True)
    tx=torch.from_numpy(x).to(device);ti=torch.from_numpy(img).to(device)
    ci=[torch.as_tensor(np.flatnonzero(labels==c),device=device) for c in names]
    weights=torch.as_tensor(trust,device=device) if trust is not None else None
    output=np.empty((len(x),len(names)),np.float32)
    weighted=np.empty_like(output) if weights is not None else None
    for start in range(0,len(x),128):
        queue.ensure_resources(deadline)
        end=min(start+128,len(x))
        sim=tx[start:end]@tx.T
        sim.masked_fill_(ti[start:end,None]==ti[None,:],-float('inf'))
        for c,indices in enumerate(ci):
            if len(indices)<5:raise ValueError('Less than five class regions')
            values,idx=sim[:,indices].topk(5,dim=1)
            if not torch.isfinite(values).all():raise ValueError('Less than five cross-image neighbors')
            output[start:end,c]=values.mean(1).cpu().numpy()
            if weights is not None:
                weighted[start:end,c]=(values-PENALTY*(1-weights[indices[idx]])).mean(1).cpu().numpy()
        if start%4096==0:print(f'[Cross-image supports] {start}/{len(x)}',flush=True)
    return output,weighted,own,names


def raw_gallery(job,keys):
    plan=ab.read(job['plan']);values={};records=ab.identities([job['plan']])
    if set(plan)!=set(job['train_ids']):raise ValueError('Training plan differs')
    for stem,rows in plan.items():
        folder=Path(job['cache'])/'train'/stem
        marker=queue.cache_read(folder,job['cache_identity'][stem])
        if marker is None:raise ValueError('Missing train vectors')
        hashes=[r['mask_sha256'] for r in rows]
        if marker['mask_hashes']!=hashes:raise ValueError('Training masks differ')
        records+=ab.identities([folder/'complete.json'])+marker['artifacts']
        with np.load(marker['file'],allow_pickle=False) as d:
            if list(d['ids'])!=[r['id'] for r in rows] or list(d['classes'])!=[r['cls'] for r in rows]:raise ValueError('Training row order differs')
            x=d['C'].copy()
        if x.shape!=(len(rows),1280) or not np.isfinite(x).all():raise ValueError('Invalid raw matrix')
        for r,v in zip(rows,x):
            key=(r['cls'],r['id'])
            if key in values:raise ValueError('Duplicate training row')
            values[key]=v
    if set(values)!=set(keys):raise ValueError('Membership differs')
    x=np.asarray([values[k] for k in keys],np.float32)
    if (np.linalg.norm(x,axis=1)<1e-6).any():raise ValueError('Zero raw vector')
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8),records


def calibration(job):
    import torch
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x,keys=provenance.gallery(job['bank'])
    labels=np.array([c for c,s in keys]);images=np.array([s.rpartition('_')[0] for c,s in keys])
    if not set(images).issubset(job['train_ids']) or set(images)&set(job['val_ids']):raise ValueError('Calibration split leakage')
    if job['stage']=='trust_prepare':
        raw,records=raw_gallery(job,keys)
        scores,_,own,names=supports(raw,labels,images,deadline=job['deadline'])
        trust=reliability(scores,own)
        dest=Path(job['attempt'])/'trust.npz'
        np.savez_compressed(dest,trust=trust,labels=labels,ids=np.array([s for c,s in keys]))
        ab.write(Path(job['attempt'])/'report.json',dict(file=str(dest),records=records,
            formula='sigmoid((same_class_top5_mean-max_other_class_top5_mean)/0.05); raw GMP; same-image excluded',
            classes={c:dict(count=int((labels==c).sum()),mean=float(trust[labels==c].mean()),q05=float(np.quantile(trust[labels==c],.05))) for c in names},
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20))
    else:
        gmp.verify(job['trust_artifacts'],'Trust artifact')
        with np.load(job['trust_file'],allow_pickle=False) as d:
            if list(zip(d['labels'],d['ids']))!=keys:raise ValueError('Trust row order changed')
            trust=d['trust'].copy()
        scores,weighted,own,names=supports(x,labels,images,trust,deadline=job['deadline'])
        report=dict(classes=names,thresholds={str(q):dict(plain=class_thresholds(scores,own,names,q),
             trust=class_thresholds(weighted,own,names,q)) for q in (.01,.05)},
             input_hashes=ab.identities(list((Path(job['bank'])/'features').glob('*/*.npz'))),
             note='In-sample learned embeddings, leave-image-out neighbors; pseudo labels only. Not calibrated error probabilities.',
             peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        ab.write(Path(job['attempt'])/'report.json',report)


def altered_predictions(query,all_feats,all_labels,index,context,topk,class_topm,search_multiplier,centroid_weight,trust,thresholds):
    from src.retrieval import _search,RetrievalPrediction
    search_k=min(max(topk,max(class_topm,1)*len(context.labels)*max(search_multiplier,1)),len(all_feats))
    sims,inds=_search(query,all_feats,index,search_k);labels=np.asarray(all_labels)
    center=query@context.centroids.T;predictions=[];decisions=[]
    for row,(ss,ii) in enumerate(zip(sims,inds)):
        valid=ii>=0;ss=ss[valid];ii=ii[valid]
        scores=np.empty(len(context.labels),np.float32);local=np.empty_like(scores);counts=[]
        for c,name in enumerate(context.labels):
            positions=np.flatnonzero(labels[ii]==name);counts.append(len(positions))
            if len(positions):
                k=min(max(class_topm,1),len(positions))
                # Use exactly the original top-m set; reliability changes its contribution only.
                chosen=positions[np.argsort(ss[positions],kind='stable')[-k:]]
                values=ss[chosen]
                if trust is not None:values=values-PENALTY*(1-trust[ii[chosen]])
                local[c]=values.mean()
            else:local[c]=center[row,c]
            scores[c]=(1-centroid_weight)*float(local[c])+centroid_weight*float(center[row,c])
        rank=np.argsort(scores)[::-1];best=int(rank[0]);label=context.labels[best]
        rejected=thresholds is not None and label!='background' and (counts[best]==0 or local[best]<thresholds[best])
        predictions.append(RetrievalPrediction('background' if rejected else label,float(scores[best]),float(scores[best]-scores[rank[1]])))
        decisions.append(dict(original_label=label,rejected=bool(rejected),local=float(local[best]),support=counts[best]))
    return predictions,decisions


def install(job):
    original=queue.install_cached_eval
    def hook(j):
        queue.KEY_IMAGES=[];original(j)
        import src.inference as inf
        enabled,q=RECIPES[job['recipe']]
        if job['recipe']=='baseline':return
        gmp.verify(job['trust_artifacts']+job['threshold_artifacts'],'Calibration inputs')
        bank,keys=provenance.gallery(job['bank'])
        with np.load(job['trust_file'],allow_pickle=False) as d:
            if list(zip(d['labels'],d['ids']))!=keys:raise ValueError('Trust order differs')
            trust=d['trust'].copy() if enabled else None
        calibration=ab.read(job['threshold_file']);original_classify=inf.classify_queries
        checked=False;stats=dict(candidates=0,rejected=0,rejected_by_class={})
        def classify(*args,**kw):
            nonlocal checked
            if args or kw['mode']!='balanced':raise ValueError('Unexpected classification protocol')
            if not checked:
                if not np.array_equal(bank,kw['all_feats']) or list(kw['all_labels'])!=[c for c,s in keys]:raise ValueError('Gallery changed')
                if list(kw['context'].labels)!=calibration['classes']:raise ValueError('Calibration classes differ')
                checked=True
            threshold=None if q is None else calibration['thresholds'][str(q)]['trust' if enabled else 'plain']
            common={k:kw[k] for k in ('all_feats','all_labels','context','topk','class_topm','search_multiplier','centroid_weight')}
            # Establish formula equivalence on every batch before applying changes.
            reference=original_classify(**kw)
            equal,_=altered_predictions(kw['query_feats'],index=kw['faiss_index'],trust=None,thresholds=None,**common)
            if any(a.label!=b.label or abs(a.score-b.score)>2e-6 or abs(a.margin-b.margin)>2e-6 for a,b in zip(reference,equal)):raise ValueError('Original formula replay differs')
            result,decisions=altered_predictions(kw['query_feats'],index=kw['faiss_index'],trust=trust,thresholds=threshold,**common)
            for d in decisions:
                stats['candidates']+=1
                if d['rejected']:
                    stats['rejected']+=1;c=d['original_label'];stats['rejected_by_class'][c]=stats['rejected_by_class'].get(c,0)+1
            return result
        inf.classify_queries=classify;job['_decision_stats']=stats
    queue.install_cached_eval=hook


def worker(job):
    if job['stage'] in ('trust_prepare','calibrate'):return calibration(job)
    install(job);old.previous.worker(job)
    ab.write(Path(job['attempt'])/'decision_stats.json',job.get('_decision_stats',dict(rejected=0)))


def summarize(results):
    groups={}
    for method in RECIPES:
        rows={str(s):results[f'{method}_{s}']['miou'] for s in old.SEEDS if f'{method}_{s}' in results}
        complete=len(rows)==3
        groups[method]=dict(complete=complete,seeds=rows,mean=statistics.mean(rows.values()) if complete else None,
            std=statistics.stdev(rows.values()) if complete else None)
        if complete and all(f'baseline_{s}' in results for s in old.SEEDS):
            groups[method]['paired_delta_pp']={str(s):100*(rows[str(s)]-results[f'baseline_{s}']['miou']) for s in old.SEEDS}
    interactions={}
    for level in ('01','05'):
        methods=('baseline','trust','reject'+level,'trust_reject'+level)
        if all(groups[m]['complete'] for m in methods):
            delta={str(s):100*(results[f'trust_reject{level}_{s}']['miou']-results[f'trust_{s}']['miou']
                -results[f'reject{level}_{s}']['miou']+results[f'baseline_{s}']['miou']) for s in old.SEEDS}
            interactions[level]=dict(by_seed_pp=delta,mean_pp=statistics.mean(delta.values()))
    return dict(groups=groups,interactions=interactions,results=results,note='VOC val development experiment; no GT in reliability/calibration, no automatic deployment; candidate rejection is not a pixel error count.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',default='bank_trust_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--worker',type=Path);p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    previous=ab.read(PREVIOUS/'manifest.json')
    if ab.read(PREVIOUS/'status.json')['status']!='complete':raise ValueError('Provenance not complete')
    gmp.verify(previous['code'],'Previous source');gmp.verify(previous['records'],'Historical inputs')
    prepared=ab.read(provenance.DEPLOY/'prepared.json')
    records=previous['records']+ab.identities([PREVIOUS/'manifest.json',PREVIOUS/'status.json'])
    identity=dict(protocol='bank_trust_v1',code=ab.source_identity(),references=records,recipes=RECIPES,tau=TAU,penalty=PENALTY,
        calibration='train pseudo labels, exclude same image, full class top5, quantiles .01/.05, original centers unchanged',
        scope='clean candidates; frozen GMP2 models; no HCR; no training GT opened',seeds=list(old.SEEDS))
    identity=old.canonical(identity)
    output=ROOT/'experiments'/args.version
    if output.exists() and (not args.resume or ab.read(output/'manifest.json')!=identity):raise ValueError('Version differs or resume required')
    queue.ensure_resources(float('inf'))
    if queue.shutil.disk_usage(ROOT).free<6*2**30:raise ValueError('Need 6 GiB free')
    print('[Preflight passed] 18 fixed full evaluations; raw trust + seed-specific training-only thresholds',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    status='running';error=None;results={};cancel=threading.Event();guard=threading.Lock();start=time.time();deadline=start+24*3600
    entry=queue.__file__;queue.__file__=__file__;stages=ab.Stages(output,True)
    saved={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Experiment interrupted')
    for s in saved:signal.signal(s,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),expected_evaluations=18))
        ab.write(output/'summary.json',dict(status=status,error=error,**summarize(results)))
    def launch(key,job,gpu):
        def action(attempt):
            job.update(attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline)
            source=prepared['jobs'][f'baseline_{job["seed"]}']
            env=os.environ.copy();env.update(source['command']['environment']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=ROOT/'logs'/args.version/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            return dict(attempt=str(attempt)),[attempt,log]
        return stages.run(key,action)
    def execute(item,gpu):
        method,seed=item;key=f'{method}_{seed}';source=prepared['jobs'][f'baseline_{seed}']
        job=copy.deepcopy(source['job']);job['recipe']=method
        if method!='baseline':
            job.update(trust_file=trust_file,trust_artifacts=trust_artifacts,threshold_file=calibrations[seed]['file'],threshold_artifacts=calibrations[seed]['artifacts'])
        value=launch(key,job,gpu);attempt=Path(value['attempt']);metrics=ab.read(attempt/'metrics.json')
        ref=ab.read(source['metrics'])
        old.validate_eval(evidence.checked_copy(metrics)[0],evidence.checked_copy(ref)[0],job['val_ids'],ref['proposal_count_after'],.2)
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metrics)
        if method=='baseline':
            old.require_image_replay(source['reference']['per_image'],str(attempt/'per_image_confusion.npz'))
            if metrics['confusion_matrix']!=ref['confusion_matrix']:raise ValueError('Baseline mismatch')
        with guard:
            results[key]=dict(miou=evidence.checked_copy(metrics)[0]['mIoU'],metrics=str(attempt/'metrics.json'),
                per_image=str(attempt/'per_image_confusion.npz'),decisions=ab.read(attempt/'decision_stats.json'))
            flush()
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        queue.run_round([('baseline',s) for s in old.SEEDS],(1,0),execute,cancel,deadline,fatal)
        job=copy.deepcopy(prepared['jobs']['baseline_42']['job']);job['stage']='trust_prepare'
        stage=launch('trust_prepare',job,1);report=ab.read(Path(stage['attempt'])/'report.json')
        trust_file=report['file'];trust_artifacts=ab.identities([trust_file,Path(stage['attempt'])/'report.json'])
        calibrations={}
        def calibrate(seed,gpu):
            job=copy.deepcopy(prepared['jobs'][f'baseline_{seed}']['job']);job.update(stage='calibrate',trust_file=trust_file,trust_artifacts=trust_artifacts)
            stage=launch(f'calibrate_{seed}',job,gpu);file=Path(stage['attempt'])/'report.json'
            with guard:calibrations[seed]=dict(file=str(file),artifacts=ab.identities([file]))
        queue.run_round(list(old.SEEDS),(1,0),calibrate,cancel,deadline,fatal)
        queue.run_round([(m,s) for m in RECIPES if m!='baseline' for s in old.SEEDS],(1,0),execute,cancel,deadline,fatal)
        gmp.verify(report['records'],'Training cache inputs');gmp.verify(records,'Inputs');gmp.verify(identity['code'],'Source')
        if len(results)!=18:raise ValueError('Incomplete evaluations')
        status='complete'
    except BaseException as exc:status='failed';error=str(exc);cancel.set();traceback.print_exc();raise
    finally:
        flush();ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(ROOT/'logs'/args.version)],read_only_history=True))
        queue.__file__=entry
        for s,h in saved.items():signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
