"""Matched-count multi-view gallery experiment; cached features, frozen adapters."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_view_augmentation_queue as views
from tools import experimental_view_hcr as hcr
ab,queue,gmp,old=views.ab,views.queue,views.gmp,views.old
METHODS=('real','duplicate','view30','view_weighted')
PROTOCOL='matched_family_multiview_gallery_v1'
RULE=dict(selection_seed=42,max_added_per_family_class='original region count',
          matched_counts='duplicate/view30/view_weighted, same family and class',
          angles={'L30':1.,'R30':1.,'L60':.25,'R60':.25},
          background='original only',instance_correspondence=False,
          clean_and_hcr=True,mlp='frozen view_weighted adapters',gt='evaluation only')


def select_rows(original, candidates):
    """Per family/class quota bounded by originals and distinct 30-degree rows.

    Same source and class identify a quota, NOT an instance correspondence.
    Sampling is fixed across model seeds. Weighted angles choose an angle first,
    then a region, without replacement. No angle is forced if detection is empty.
    """
    selected={m:[] for m in METHODS[1:]}; quotas=[]
    for key in sorted(original):
        family,cls=key
        if cls=='background':continue
        real=original[key]; available=candidates.get(key,[])
        thirty=[r for r in available if r['view'] in ('L30','R30')]
        count=min(len(real),len(thirty))
        if not count:continue
        seed=int.from_bytes(hashlib.sha256((family+'|'+cls+'|42').encode()).digest()[:8],'big')
        rng=np.random.default_rng(seed)
        dup=rng.choice(len(real),count,replace=False)
        selected['duplicate'] += [dict(family=family,cls=cls,real_row=int(real[int(i)])) for i in dup]
        chosen=rng.choice(len(thirty),count,replace=False)
        selected['view30'] += [dict(thirty[int(i)]) for i in chosen]
        pool=list(available)
        for _ in range(count):
            angles=[a for a in views.VIEWS if any(r['view']==a for r in pool)]
            weights=np.asarray([RULE['angles'][a] for a in angles]);weights/=weights.sum()
            angle=str(rng.choice(angles,p=weights))
            indices=[i for i,r in enumerate(pool) if r['view']==angle]
            selected['view_weighted'].append(pool.pop(int(rng.choice(indices))))
        quotas.append(dict(family=family,cls=cls,original=len(real),added=count))
    if not quotas:raise ValueError('No eligible cached foreground family/class')
    return dict(protocol=PROTOCOL,rule=RULE,rows=selected,quotas=quotas,
                total_added=sum(r['added'] for r in quotas))


def make_plan(job):
    from src.feature_bank_storage import load_feature_matrix_npz
    from src.adapter_training import image_id_from_sample
    original=defaultdict(list);candidates=defaultdict(list)
    for directory in sorted((Path(job['bank'])/'features').iterdir()):
        if not directory.is_dir():continue
        files=list(directory.glob('*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous original class bank')
        features,ids=load_feature_matrix_npz(files[0])
        for i,sample in enumerate(ids):original[image_id_from_sample(sample),directory.name].append(i)
    permitted=set(job['split']['train_images'])
    forbidden=set(job['split']['validation_images'])|set(job['val_ids'])
    for marker in sorted(Path(job['view_cache']).glob('*/complete.json')):
        entry=ab.read(marker);identity=entry['identity'];source=identity['input'];family=source['source_id']
        if family not in permitted or family in forbidden:raise ValueError('Synthetic family split leakage')
        if identity['signature']!=job['raw_signature'] or identity['rule']!=views.RULE:
            raise ValueError('Cached feature algorithm changed')
        gmp.verify(entry['artifacts'],'Generated view cache')
        with np.load(entry['file'],allow_pickle=False) as f:
            x=f['vectors'];labels=f['labels']
            if x.shape!=(len(labels),1280) or not np.isfinite(x).all():raise ValueError('Invalid cached vectors')
            if len(x) and (np.linalg.norm(x,axis=1)<1e-6).any():raise ValueError('Zero cached foreground')
            for row,c in enumerate(labels):
                if int(c) not in range(1,21):raise ValueError('Synthetic background/disallowed class')
                cls=views.provenance.evidence.VOC_CLASSES[int(c)]
                candidates[family,cls].append(dict(family=family,cls=cls,view=source['view'],
                    key=marker.parent.name,file=entry['file'],row=row,marker=str(marker)))
    return select_rows(original,candidates)


def build_banks(job):
    import torch
    from src.feature_bank_storage import load_feature_matrix_npz,save_feature_matrix_npz
    old.previous.mlp.install_adapter('d2')
    from src.feature_adapter import load_adapter,project_features,adapted_signature
    adapter,_=load_adapter(Path(job['weights'])/'adapter.pt','cuda:0',job['raw_signature'])
    expected=adapted_signature(job['raw_signature'],Path(job['weights'])/'adapter.pt',adapter.architecture)
    if expected!=ab.read(Path(job['bank'])/'meta/feature_signature.json'):raise ValueError('Original bank/adapter mismatch')
    plan=ab.read(job['plan_file'])
    if ab.digest(job['plan_file'])!=job['plan_sha256'] or plan['rule']!=RULE:raise ValueError('Gallery plan changed')
    cache={};extra={m:defaultdict(list) for m in METHODS[1:]};extra_ids={m:defaultdict(list) for m in METHODS[1:]}
    originals={}
    for directory in sorted((Path(job['bank'])/'features').iterdir()):
        if directory.is_dir():
            file=list(directory.glob('*.npz'))
            if len(file)!=1:raise ValueError('Ambiguous original class matrix')
            originals[directory.name]=(file[0],*load_feature_matrix_npz(file[0]))
    for method in METHODS[1:]:
        raw=[];raw_rows=[]
        for n,row in enumerate(plan['rows'][method]):
            cls=row['cls']
            if method=='duplicate':value=originals[cls][1][row['real_row']]
            else:
                if row['file'] not in cache:
                    entry=ab.read(row['marker']);gmp.verify(entry['artifacts'],'Selected synthetic rows')
                    with np.load(row['file'],allow_pickle=False) as f:cache[row['file']]=(f['vectors'].copy(),f['labels'].copy())
                x,labels=cache[row['file']]
                if views.provenance.evidence.VOC_CLASSES[int(labels[row['row']])]!=cls:raise ValueError('Selected label/row changed')
                raw.append(x[row['row']]);raw_rows.append((cls,n));continue
            extra[method][cls].append(value)
            extra_ids[method][cls].append(f"{row['family']}_mv_{method}_{n:06d}")
        if raw:
            projected=project_features(adapter,np.asarray(raw,np.float32),'cuda:0')
            for (cls,n),value,row in zip(raw_rows,projected,plan['rows'][method]):
                extra[method][cls].append(value)
                extra_ids[method][cls].append(f"{row['family']}_mv_{method}_{n:06d}")
        bank=Path(job['bank_root'])/method/job['attempt_name']
        bank.mkdir(parents=True,exist_ok=False);(bank/'meta').mkdir()
        for name in ('class_mapping.json','feature_signature.json'):
            shutil.copy2(Path(job['bank'])/'meta'/name,bank/'meta'/name)
        counts={}
        for cls,(file,x,ids) in originals.items():
            added=np.asarray(extra[method][cls],np.float32).reshape(-1,256)
            all_x=np.concatenate([x,added]);all_ids=[*ids.tolist(),*extra_ids[method][cls]]
            if not np.isfinite(all_x).all() or (np.linalg.norm(all_x,axis=1)<1e-6).any():raise ValueError('Invalid projected gallery')
            save_feature_matrix_npz(bank/file.relative_to(job['bank']),all_x,all_ids)
            counts[cls]=dict(original=len(x),added=len(added),total=len(all_x))
        ab.write(bank/'meta/multiview_gallery.json',dict(protocol=PROTOCOL,rule=RULE,method=method,
            plan_sha256=job['plan_sha256'],adapter_sha256=ab.digest(Path(job['weights'])/'adapter.pt'),counts=counts))
        print('[Bank]',method,'added',sum(v['added'] for v in counts.values()),flush=True)
    # All controls have exactly equal class counts, not just equal totals.
    values=[ab.read(Path(job['bank_root'])/m/job['attempt_name']/'meta/multiview_gallery.json')['counts'] for m in METHODS[1:]]
    if any(v!=values[0] for v in values[1:]):raise ValueError('Class count matching failed')


def worker(job):
    import torch
    started=time.monotonic();status='failed';error=None;category='local'
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if job['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
        if job['stage']=='build':build_banks(job)
        elif job['stage']=='eval':
            if job['hcr']:
                install=queue.install_cached_eval
                def relocated(j):
                    install(j)
                    import run_inference as inference
                    run=inference.main
                    inference.main=lambda argv:run([*argv,'--rescue-raw-dir',job['raw_proposals']])
                queue.install_cached_eval=relocated
                hcr.validator.worker(job)
            else:queue.KEY_IMAGES=[];old.previous.worker(job)
        else:raise ValueError('No training/backbone extraction allowed')
        torch.cuda.synchronize();status='complete'
    except queue.Deferred as exc:category='deferred';error=str(exc);raise
    except torch.cuda.OutOfMemoryError as exc:category='oom';error=str(exc);raise
    except (ValueError,KeyError,FileNotFoundError,queue.GlobalFailure) as exc:category='global';error=str(exc);raise
    except BaseException as exc:error=str(exc);raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,error=error,category=category,
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def summarize(results):
    groups={};contrasts={}
    for rescue in (False,True):
        suffix='hcr' if rescue else 'clean'
        for method in METHODS:
            key=method+'_'+suffix
            scores={str(s):results[f'{key}_{s}']['miou'] for s in old.SEEDS if f'{key}_{s}' in results}
            groups[key]=dict(complete=len(scores)==3,seeds=scores,mean=float(np.mean(list(scores.values()))) if len(scores)==3 else None,
                sample_std=float(np.std(list(scores.values()),ddof=1)) if len(scores)==3 else None)
        for a,b in [('duplicate','real'),('view30','duplicate'),('view_weighted','duplicate'),('view_weighted','view30'),('view30','real'),('view_weighted','real')]:
            if groups[a+'_'+suffix]['complete'] and groups[b+'_'+suffix]['complete']:
                delta={s:100*(groups[a+'_'+suffix]['seeds'][s]-groups[b+'_'+suffix]['seeds'][s]) for s in groups[a+'_'+suffix]['seeds']}
                contrasts[a+'-'+b+'_'+suffix]=dict(by_seed_pp=delta,mean_pp=float(np.mean(list(delta.values()))))
    return dict(groups=groups,contrasts=contrasts,results=results,note='Fixed development batch; no instance matching/GT selection/significance/automatic deployment.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--version',default='multiview_bank_20261008')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=24)
    args=p.parse_args();os.chdir(ROOT);views.install_relocation()
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    if not np.isfinite(args.max_hours) or args.max_hours<=0:raise ValueError('Invalid time budget')
    source=ROOT/'experiments/view_aug_20261008_checked';hs=ROOT/'experiments/view_hcr_20261008'
    summary=ab.read(source/'summary.json');hsummary=ab.read(hs/'summary.json')
    if summary['status']!='complete' or hsummary['status']!='complete':raise ValueError('Previous experiments incomplete')
    manifest=ab.read(source/'manifest.json');gmp.verify(views.unique_records(manifest['code']),'Frozen source')
    records=views.unique_records(manifest['records']);gmp.verify(records,'Historical inputs')
    jobs={};envs={};refs={}
    for s in old.SEEDS:
        r=hsummary['results'][f'weighted_{s}'];attempt=Path(r['metrics']).parent
        job=ab.read(attempt/'job.json');env=ab.read(attempt/'command.json')['environment']
        bank=summary['results'][f'view_weighted_{s}']['training']['bank']
        if job['bank']!=bank:raise ValueError('Models differ between clean/HCR')
        jobs[s]=job;envs[s]=env;refs[s]={False:summary['results'][f'view_weighted_{s}'],True:r}
        for kind in ('train','eval'):
            marker=source/f'{kind}_view_weighted_{s}'/'complete.json';stage=ab.read(marker)
            gmp.verify(views.unique_records(stage['artifacts']),'Previous model/evaluation');records+=stage['artifacts']+ab.identities([marker])
        marker=hs/f'weighted_{s}'/'complete.json';stage=ab.read(marker)
        gmp.verify(views.unique_records(stage['artifacts']),'HCR reference');records+=stage['artifacts']+ab.identities([marker])
    for lane in (0,1):
        marker=source/f'extract_{lane}'/'complete.json';stage=ab.read(marker)
        print('[Preflight] cached view shard',lane,flush=True)
        gmp.verify(views.unique_records(stage['artifacts']),'Generated masks/vectors');records+=stage['artifacts']+ab.identities([marker])
    cache=ROOT/'storage/raw_banks/voc/view_aug_20261008_checked/generated_regions'
    if len(list(cache.glob('*/complete.json')))!=33864:raise ValueError('Incomplete view cache')
    records=views.unique_records(records);code=ab.source_identity()
    identity=old.canonical(dict(protocol=PROTOCOL,rule=RULE,records=records,code=code,max_hours=args.max_hours))
    output=ROOT/'experiments'/args.version;bank_root=ROOT/'storage/mlp_banks/voc'/args.version;logs=ROOT/'logs'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Existing version differs/needs resume')
    else:
        if bank_root.exists() or logs.exists():raise FileExistsError('New version paths already used')
        ab.write(output/'manifest.json',identity)
    for gpu in (0,1):ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    if shutil.disk_usage(ROOT).free<queue.FLOOR+3*2**30:raise queue.Deferred('Require 3 GiB allowance plus 5 GiB reserve')
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    stages=views.AliasAwareStages(output,True);cancel=threading.Event();guard=threading.Lock()
    results={};failures={};status='running';error=None;start=time.time();deadline=start+args.max_hours*3600
    queue.__file__=__file__
    def stop(*_):cancel.set();raise KeyboardInterrupt('Stopped')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),failures=failures,expected=24))
        ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,**summarize(results)))
    def launch(key,job,gpu):
        def action(attempt):
            job.update(protocol=PROTOCOL,attempt=str(attempt),attempt_name=attempt.name,deadline=deadline,metrics=str(attempt/'metrics.json'))
            artifacts=[attempt];result=dict(attempt=str(attempt))
            if job['stage']=='build':
                job.update(bank_root=str(bank_root/str(job['seed'])),plan_file=str(output/'selection.json'),plan_sha256=ab.digest(output/'selection.json'))
                result['banks']={m:str(Path(job['bank_root'])/m/attempt.name) for m in METHODS[1:]};artifacts+=list(result['banks'].values())
            env=os.environ.copy();env.update(envs[job['seed']]);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env=views.evaluation_environment(job,env);env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=logs/f'{key}_{attempt.name}.log';queue.launch(job,attempt,env,log,cancel,deadline)
            return result,[*artifacts,log]
        return stages.run(key,action)
    def evaluate(item,gpu):
        method,rescue,s=item;key=f'{method}_{"hcr" if rescue else "clean"}_{s}'
        job=copy.deepcopy(jobs[s]);job.update(stage='eval',hcr=rescue)
        if method!='real':job['bank']=built[s]['banks'][method]
        r=launch(key,job,gpu);attempt=Path(r['attempt']);metric=ab.read(attempt/'metrics.json')
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
        with np.load(attempt/'per_image_confusion.npz',allow_pickle=False) as f:
            if len(f['ids'])!=1449 or f['ids'].tolist()!=job['val_ids']:raise ValueError('Validation IDs differ')
        if metric['feature_signature']!=ab.read(Path(job['bank'])/'meta/feature_signature.json'):raise ValueError('Signature mismatch')
        if rescue:
            if metric['semantic_rescue']['parameters']!=hcr.PARAMS:raise ValueError('HCR parameters changed')
        elif 'semantic_rescue' in metric:raise ValueError('Unexpected rescue')
        if method=='real':
            old.require_image_replay(refs[s][rescue]['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=ab.read(refs[s][rescue]['metrics'])['confusion_matrix']:raise ValueError('Historical replay differs')
        corrected=views.provenance.evidence.checked_copy(metric)[0]
        with guard:
            results[key]=dict(miou=corrected['mIoU'],per_class_iou=corrected['per_class_iou'],metrics=str(attempt/'metrics.json'),per_image=str(attempt/'per_image_confusion.npz'),bank=job['bank']);flush()
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        built={}
        queue.run_round([('real',r,s) for s in old.SEEDS for r in (False,True)],(0,1),evaluate,cancel,deadline,fatal)
        def plan_action(attempt):
            job=dict(jobs[42],split=manifest['split'],view_cache=str(cache));plan=make_plan(job)
            ab.write(output/'selection.json',plan)
            return dict(added=plan['total_added']),[output/'selection.json']
        stages.run('selection',plan_action)
        def build(s,gpu):built[s]=launch(f'build_{s}',dict(jobs[s],stage='build'),gpu)
        queue.run_round(list(old.SEEDS),(0,1),build,cancel,deadline,fatal)
        def failed(item,exc):
            with guard:failures[str(item)]=str(exc);flush()
        queue.run_round([(m,r,s) for s in old.SEEDS for m in METHODS[1:] for r in (False,True)],(0,1),evaluate,cancel,deadline,failed)
        gmp.verify(code,'Frozen code');stages.verify_all();status='complete' if len(results)==24 else 'partial'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();lock.unlink();ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(bank_root),str(logs)],automatic_deletion=False))
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
