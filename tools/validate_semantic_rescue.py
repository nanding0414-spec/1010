"""Deployment regression for HCR. Historical caches supply features only.

Prepare before replacing production files; run after deploying reviewed files.
All candidate loading, classification, NMS, rescue and rasterization use mainline.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_rescue_evidence as history
ab,queue,gmp,old=history.ab,history.queue,history.gmp,history.old
CHANGED=('run_inference.py','src/inference.py')
ADDED=('src/semantic_rescue.py',)


def prepare(folder):
    if folder.exists():raise FileExistsError(folder)
    previous=ROOT/'experiments/rescue_evidence_20261004_181543'
    manifest=ab.read(previous/'manifest.json')
    if ab.read(previous/'status.json')['status']!='complete':raise ValueError('History incomplete')
    gmp.verify(manifest['code'],'Original code');gmp.verify(manifest['references'],'Original inputs')
    records=list(manifest['references'])+list(manifest['code'])
    records+=ab.identities([previous/'manifest.json',previous/'status.json'])
    jobs={}
    for method in ('baseline','high95_only'):
        for seed in old.SEEDS:
            key=f'{method}_{seed}';ref=queue.verified_stage(previous,key,records)
            attempt=Path(ref['report']).parent
            records+=ab.identities([previous/key/'complete.json'])
            jobs[key]=dict(job=ab.read(attempt/'job.json'),command=ab.read(attempt/'command.json'),reference=ref,
                           metrics=str(attempt/'metrics.json'))
    code=ab.source_identity()
    folder.mkdir(parents=True)
    for record in code:
        src=Path(record['path']);dest=folder/'source_before'/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    ab.write(folder/'prepared.json',dict(previous=str(previous),code_before=code,records=records,jobs=jobs))
    print('[Prepared]',folder,flush=True)


def worker(job):
    original=queue.install_cached_eval
    def install(j):
        queue.KEY_IMAGES=[]
        original(j)
        import run_inference as inference
        read=queue.cache_vectors
        def vectors(folder,identity,group,hashes):
            if not job['hcr']:return read(folder,identity,group,hashes)
            stem=Path(folder).name
            strong,_,extras,_,_=history.prior.pool(job,stem)
            strong_hashes=queue.mask_hashes(strong);extra_hashes=queue.mask_hashes(extras)
            if hashes!=strong_hashes+extra_hashes:raise ValueError('Mainline HCR candidate pool/order differs from experiment')
            added,stored=history.prior.cached_extra(job,stem)
            if stored!=extra_hashes:raise ValueError('Extra feature identities differ')
            return np.concatenate([read(folder,identity,group,strong_hashes),added],axis=0)
        queue.cache_vectors=vectors
        main=inference.main
        if job['hcr']:inference.main=lambda argv: main([*argv,'--rescue'])
    queue.install_cached_eval=install
    # Experimental signature/adapter compatibility is limited to this validator.
    # The mainline's signature rules are not weakened for general inference.
    old.previous.worker(job)


def run(folder):
    prepared=ab.read(folder/'prepared.json')
    before={r['path']:r for r in prepared['code_before']}
    after=ab.source_identity();current={r['path']:r for r in after}
    allowed={str((ROOT/p).resolve()) for p in CHANGED}
    added={str((ROOT/p).resolve()) for p in ADDED}
    for path,record in before.items():
        if current.get(path)!=record and path not in allowed:raise ValueError('Unreviewed production change: '+path)
    if set(current)-set(before)!=added:raise ValueError('Unexpected new source files')
    for path in allowed:
        if path not in current:raise ValueError('Missing deployed file')
        backup=folder/'source_before'/Path(path).relative_to(ROOT)
        if ab.digest(backup)!=before[path]['sha256']:raise ValueError('Original source backup changed')
    retained=[r for r in prepared['records'] if r['path'] not in allowed]
    gmp.verify(retained,'Retained inputs and code')
    deployment=folder/'deployment.json'
    expected=dict(changed=list(CHANGED),added=list(ADDED),code_after=after)
    if deployment.exists() and ab.read(deployment)!=expected:raise ValueError('Deployment identity changed')
    if not deployment.exists():ab.write(deployment,expected)
    for gpu in (0,1):ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    stages=ab.Stages(folder/'validation',True);stages.verify_all()
    lock=folder/'validation.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    cancel=threading.Event();guard=threading.Lock();results={};status='running';error=None
    start=time.time();deadline=start+12*3600
    old_entry=queue.__file__;queue.__file__=__file__
    saved={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Validation interrupted')
    for s in saved:signal.signal(s,stop)
    def flush():
        ab.write(folder/'status.json',dict(status=status,error=error,completed=sorted(results)))
        ab.write(folder/'summary.json',dict(status=status,results=results,expected_cells=6,
            note='Full mainline inference replay with historical pre-MLP feature caches; no new SAM3 feature extraction.'))
    def execute(item,gpu):
        method,seed=item;key=f'{method}_{seed}';source=prepared['jobs'][key]
        def action(attempt):
            job=copy.deepcopy(source['job']);job.update(attempt=str(attempt),metrics=str(attempt/'metrics.json'),
                                                       hcr=method=='high95_only',deadline=deadline)
            env=os.environ.copy();env.update(source['command']['environment']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            log=ROOT/'logs'/folder.name/(key+'_'+attempt.name+'.log')
            queue.launch(job,attempt,env,log,cancel,deadline)
            metric=ab.read(job['metrics']);reference=ab.read(source['metrics'])
            count=ab.read(prepared['jobs'][f'baseline_{seed}']['metrics'])['proposal_count_after']
            reference['proposal_count_before']=reference['proposal_count_after']=count
            old.validate_eval(history.checked_copy(metric)[0],history.checked_copy(reference)[0],job['val_ids'],count,.2)
            old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
            old.require_image_replay(source['reference']['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=reference['confusion_matrix']:raise ValueError('Mainline full confusion differs')
            if metric['feature_signature']!=ab.read(Path(job['bank'])/'meta/feature_signature.json'):raise ValueError('Bank signature changed')
            if job['hcr']:
                recorded=metric['semantic_rescue']
                if recorded['parameters']!=dict(proposal_score=.4,retrieval_score=.95,margin=.1,novel_fraction=.1):raise ValueError('Wrong HCR defaults')
                if len(recorded['images'])!=1449:raise ValueError('Incomplete rescue diagnostics')
            elif 'semantic_rescue' in metric:raise ValueError('Disabled HCR changed baseline')
            return dict(metrics=str(attempt/'metrics.json'),per_image=str(attempt/'per_image_confusion.npz'),
                        mIoU=history.checked_copy(metric)[0]['mIoU'],identical_to_history=True),[attempt,log]
        try:value=stages.run(key,action)
        except BaseException:traceback.print_exc();raise
        with guard:results[key]=value;flush()
    def fatal(item,exc):raise exc
    try:
        flush()
        queue.run_round([(m,s) for m in ('baseline','high95_only') for s in old.SEEDS],(0,1),execute,cancel,deadline,fatal)
        gmp.verify(retained,'Retained inputs and code')
        if ab.source_identity()!=after:raise ValueError('Source changed during regression')
        status='complete'
    except BaseException as exc:
        status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();queue.__file__=old_entry
        for s,h in saved.items():signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(folder.name,status,str(folder),time.time()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version');p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path)
    args=p.parse_args();os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    if not args.version:p.error('--version required')
    ab.parse_args(['--version',args.version]);folder=ROOT/'experiments'/args.version
    if args.prepare:prepare(folder)
    else:run(folder)


if __name__=='__main__':main()
