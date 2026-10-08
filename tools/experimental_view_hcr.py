"""Fixed HCR transfer to completed angle-weighted adapters; evaluation only."""
import argparse
import copy
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
from tools import validate_semantic_rescue as validator
ab,queue,gmp,old=views.ab,views.queue,views.gmp,views.old
PARAMS=dict(proposal_score=.4,retrieval_score=.95,margin=.1,novel_fraction=.1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path)
    p.add_argument('--version',required=False,default='view_hcr_20261008')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();os.chdir(ROOT);views.install_relocation()
    if args.worker:
        job=ab.read(args.worker)
        if job.get('hcr') is not True or job['stage']!='eval':raise ValueError('HCR evaluation only')
        install=queue.install_cached_eval
        def relocated_install(j):
            install(j)
            import run_inference as inference
            run=inference.main
            inference.main=lambda argv:run([*argv,'--rescue-raw-dir',job['raw_proposals']])
        queue.install_cached_eval=relocated_install
        return validator.worker(job)
    ab.parse_args(['--version',args.version])
    source=ROOT/'experiments/view_aug_20261008_checked'
    deploy=views.provenance.DEPLOY
    manifest=ab.read(source/'manifest.json'); summary=ab.read(source/'summary.json')
    if summary['status']!='complete' or not summary['groups']['view_weighted']['complete']:raise ValueError('Need complete three-seed augmentation')
    print('[Preflight] verifying frozen code and historical inputs',flush=True)
    gmp.verify(views.unique_records(manifest['code']),'Frozen augmentation code')
    records=views.unique_records(manifest['records']);gmp.verify(records,'Historical inputs')
    historical=ab.read(deploy/'summary.json')
    if historical['status']!='complete':raise ValueError('Incomplete HCR reference')
    jobs={};refs={};environments={}
    for seed in old.SEEDS:
        folder=deploy/f'validation/high95_only_{seed}/attempt_001'
        job=ab.read(folder/'job.json');env=ab.read(folder/'command.json')['environment']
        if not job['hcr'] or job['role']!='gmp2' or job['group']!='C':raise ValueError('Wrong HCR reference')
        refs[seed]=historical['results'][f'high95_only_{seed}']
        metric=ab.read(refs[seed]['metrics'])
        if metric['semantic_rescue']['parameters']!=PARAMS:raise ValueError('Historical HCR parameters differ')
        jobs['baseline',seed]=job;environments[seed]=env
        marker=source/f'train_view_weighted_{seed}/complete.json'
        stage=ab.read(marker);gmp.verify(views.unique_records(stage['artifacts']),'Weighted model')
        trained=stage['result']
        candidate=copy.deepcopy(job);candidate.update(weights=trained['weights'],bank=trained['bank'])
        if ab.read(Path(candidate['bank'])/'meta/feature_signature.json')['adapter']['sha256']!=ab.digest(Path(candidate['weights'])/'adapter.pt'):
            raise ValueError('Adapter/bank identity differs')
        jobs['weighted',seed]=candidate
        records+=stage['artifacts']+ab.identities([marker,folder/'job.json',folder/'command.json',Path(refs[seed]['metrics']),Path(refs[seed]['per_image'])])
    records=views.unique_records(records)
    code=ab.source_identity();identity=old.canonical(dict(protocol='fixed_view_hcr_v1',code=code,records=records,parameters=PARAMS))
    output=ROOT/'experiments'/args.version
    if output.exists():
        if not args.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Existing version differs/needs resume')
    else:ab.write(output/'manifest.json',identity)
    for gpu in (0,1):ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    stages=views.AliasAwareStages(output,True);cancel=threading.Event();guard=threading.Lock()
    start=time.time();deadline=start+12*3600;results={};status='running';error=None
    queue.__file__=__file__
    def stop(*_):cancel.set();raise KeyboardInterrupt('Stopped')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def flush():
        groups={}
        for name in ('baseline','weighted'):
            scores={str(s):results[f'{name}_{s}']['miou'] for s in old.SEEDS if f'{name}_{s}' in results}
            groups[name]=dict(seeds=scores,complete=len(scores)==3,mean=float(np.mean(list(scores.values()))) if len(scores)==3 else None)
        delta={str(s):100*(results[f'weighted_{s}']['miou']-results[f'baseline_{s}']['miou']) for s in old.SEEDS if f'weighted_{s}' in results and f'baseline_{s}' in results}
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),expected=6))
        ab.write(output/'summary.json',dict(status=status,error=error,groups=groups,paired_delta_pp=delta,
            mean_delta_pp=float(np.mean(list(delta.values()))) if len(delta)==3 else None,results=results,
            note='Fixed HCR; VOC val development only; no automatic deployment.'))
    def evaluate(item,gpu):
        method,seed=item;key=f'{method}_{seed}'
        def action(attempt):
            job=copy.deepcopy(jobs[item]);job.update(attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline)
            env=os.environ.copy();env.update(environments[seed]);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env=views.evaluation_environment(job,env)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=ROOT/'logs'/args.version/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            metric=ab.read(job['metrics']);reference=ab.read(refs[seed]['metrics'])
            if metric['semantic_rescue']['parameters']!=PARAMS or len(metric['semantic_rescue']['images'])!=1449:raise ValueError('HCR changed/incomplete')
            old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
            with np.load(attempt/'per_image_confusion.npz',allow_pickle=False) as f:
                if f['ids'].tolist()!=job['val_ids'] or len(f['ids'])!=1449:raise ValueError('Validation IDs differ')
            if method=='baseline':
                old.require_image_replay(refs[seed]['per_image'],str(attempt/'per_image_confusion.npz'))
                if metric['confusion_matrix']!=reference['confusion_matrix']:raise ValueError('Historical HCR replay differs')
            if metric['feature_signature']!=ab.read(Path(job['bank'])/'meta/feature_signature.json'):raise ValueError('Signature mismatch')
            corrected=views.provenance.evidence.checked_copy(metric)[0]
            return dict(miou=corrected['mIoU'],metrics=job['metrics'],per_image=str(attempt/'per_image_confusion.npz'),
                per_class_iou=corrected['per_class_iou']),[attempt,log]
        value=stages.run(key,action)
        with guard:results[key]=value;flush()
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        for method in ('baseline','weighted'):
            queue.run_round([(method,s) for s in old.SEEDS],(0,1),evaluate,cancel,deadline,fatal)
        gmp.verify(code,'Frozen code');stages.verify_all();status='complete'
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();lock.unlink()
        ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(ROOT/'logs'/args.version)],automatic_deletion=False))
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
