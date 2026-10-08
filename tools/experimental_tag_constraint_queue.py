"""Train-side image-tag audit and fixed absent-class-loss control experiment.

Three seeds, original GMP2 hard bank, identical trainer except one CE addend.
No segmentation GT is opened by the training/audit code.
"""
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
HISTORY=ROOT/'experiments/bank_trust_20261005_112906'
TRAIN_HISTORY=ROOT/'experiments/gmp_mlp2_20261002_141439'
METHODS=('historical','control','absent01')
COEFFICIENT=.1


def parse_tags(data, image_ids, mapping):
    if not isinstance(data,dict) or not isinstance(data.get('images'),list):raise ValueError('Expected historical image-level JSON schema')
    tags={}
    for row in data['images']:
        stem=str(row['img_id'])
        if stem in tags:raise ValueError('Duplicate image tag row')
        values=row['labels']
        if not isinstance(values,list) or any(type(x) is not int or str(x) not in mapping for x in values):raise ValueError('Invalid image-level category')
        tags[stem]={mapping[str(v)] for v in values}|{'background'}
    if set(image_ids)-set(tags):raise ValueError('Missing training image tags')
    return {stem:tags[stem] for stem in image_ids}


def allowed_rows(groups,classes,tags):
    if 'background' not in classes or len(set(classes))!=len(classes):raise ValueError('Invalid class order')
    return np.asarray([[c in tags[str(stem)] for c in classes] for stem in groups],bool)


def coverage(groups,y,classes,tags):
    present={stem:set() for stem in tags}
    for stem,label in zip(groups,y):present[str(stem)].add(classes[int(label)])
    missing={stem:sorted((tags[stem]-{'background'})-present[stem]) for stem in tags}
    conflicts=[dict(image=str(stem),pseudo_class=classes[int(label)]) for stem,label in zip(groups,y) if classes[int(label)] not in tags[str(stem)]]
    return dict(images=len(tags),regions=len(y),images_with_missing_positive=sum(bool(v) for v in missing.values()),
                missing_positive_by_image={k:v for k,v in missing.items() if v},pseudo_class_outside_tags=conflicts,
                note='Missing region evidence is not proof of a missing object; merged-background vector may contain it. No GT used.')


def absent_loss(logits,allowed):
    """-log P(any allowed class), stable at extreme logits; background allowed."""
    import torch
    if logits.ndim!=2 or allowed.shape!=logits.shape or allowed.dtype!=torch.bool or not allowed.any(1).all():raise ValueError('Invalid allowed class mask')
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite classifier logits')
    return (torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~allowed,-torch.inf),dim=1)).mean()


def classifier_audit(model,head,x,groups,y,classes,tags,train,val,device):
    import torch
    from src.feature_adapter import project_features
    z=project_features(model,x,device);allowed=allowed_rows(groups,classes,tags)
    all_mass=[];forbidden_top=[]
    head.eval()
    with torch.inference_mode():
        for start in range(0,len(z),512):
            logits=head(torch.from_numpy(z[start:start+512]).to(device));p=logits.softmax(1).cpu().numpy()
            mask=allowed[start:start+512]
            all_mass.extend((p*(~mask)).sum(1).tolist())
            forbidden_top.extend((~mask[np.arange(len(p)),p.argmax(1)]).tolist())
    mass=np.asarray(all_mass);forbidden=np.asarray(forbidden_top)
    def report(rows):
        return dict(regions=len(rows),mean_absent_probability=float(mass[rows].mean()),
            absent_top1_count=int(forbidden[rows].sum()),absent_top1_fraction=float(forbidden[rows].mean()),
            by_pseudo_class={c:dict(count=int((y[rows]==i).sum()),mean_absent_probability=float(mass[rows[y[rows]==i]].mean()))
                for i,c in enumerate(classes) if (y[rows]==i).any()})
    return dict(training=report(train),internal_validation=report(val),
                note='Temporary classification head probabilities on held-out TRAIN images; not retrieval mIoU or GT accuracy.')


def install_training(job):
    import torch
    from torch import nn
    from torch.nn import functional as F
    import src.adapter_training as at
    original_load,original_batch=at.load_training_bank,at.balanced_batch
    original_ce,original_init,original_save=F.cross_entropy,nn.Linear.__init__,torch.save
    state=dict(batches=0,ce_sum=0.,absent_sum=0.,train_absent_mass_sum=0.)
    def load(*args,**kwargs):
        data=original_load(*args,**kwargs);x,y,groups,classes,sig,records=data
        mapping=ab.read(Path(job['raw'])/'meta/class_mapping.json')['id_to_dirname']
        tags=parse_tags(ab.read(job['imagelevel_json']),job['train_ids'],mapping)
        mask=allowed_rows(groups,classes,tags)
        if not mask[np.arange(len(y)),y].all():raise ValueError('Pseudo class outside image labels')
        state.update(x=x,y=y,groups=groups,classes=classes,tags=tags,allowed=mask)
        ab.write(Path(job['attempt'])/'tag_coverage.json',coverage(groups,y,classes,tags))
        return data
    def batch(*args,**kwargs):
        rows=original_batch(*args,**kwargs);state['rows']=rows.copy();return rows
    def init(layer,in_features,out_features,*args,**kwargs):
        original_init(layer,in_features,out_features,*args,**kwargs)
        if in_features==256 and out_features==len(state.get('classes',[])):
            if 'head' in state:raise ValueError('Multiple classifier heads')
            state['head']=layer
    def ce(logits,target,*args,**kwargs):
        base=original_ce(logits,target,*args,**kwargs)
        rows=state['rows']
        if len(target)!=len(rows) or not np.array_equal(target.detach().cpu().numpy(),state['y'][rows]):raise ValueError('Training row/tag misalignment')
        mask=torch.as_tensor(state['allowed'][rows],device=logits.device)
        aux=absent_loss(logits,mask)
        state['batches']+=1;state['ce_sum']+=float(base.detach());state['absent_sum']+=float(aux.detach())
        state['train_absent_mass_sum']+=float((logits.detach().softmax(1)*(~mask)).sum(1).mean())
        # Exact control path: no added zero operations in the optimization graph.
        return base if job['coefficient']==0 else base+job['coefficient']*aux
    def save(payload,path,*args,**kwargs):
        result=original_save(payload,path,*args,**kwargs)
        if isinstance(payload,dict) and payload.get('format_version')==1 and 'source_signature' in payload:
            sidecar=Path(job['weights'])/'classifier.pt'
            temp=sidecar.with_suffix('.tmp')
            original_save(dict(epoch=payload['epoch'],classes=state['classes'],
                state_dict={k:v.detach().cpu() for k,v in state['head'].state_dict().items()},
                coefficient=job['coefficient'],image_tags_sha256=ab.digest(job['imagelevel_json'])),temp)
            os.replace(temp,sidecar)
        return result
    at.load_training_bank=load;at.balanced_batch=batch;nn.Linear.__init__=init;F.cross_entropy=ce;torch.save=save
    return state,(original_load,original_batch,original_init,original_ce,original_save)


def train_worker(job):
    import torch
    from torch import nn
    from torch.nn import functional as F
    import src.adapter_training as at
    state,original=install_training(job)
    try:old.previous.worker(job)
    finally:at.load_training_bank,at.balanced_batch,nn.Linear.__init__,F.cross_entropy,torch.save=original
    from src.feature_adapter import load_adapter
    adapter,payload=load_adapter(Path(job['weights'])/'adapter.pt','cuda:0',job['raw_signature'])
    stored=torch.load(Path(job['weights'])/'classifier.pt',weights_only=True,map_location='cpu')
    report=ab.read(Path(job['weights'])/'training_report.json')
    if stored['epoch']!=payload['epoch'] or stored['epoch']!=report['best_epoch'] or stored['classes']!=state['classes']:raise ValueError('Classifier checkpoint not paired to best MLP')
    head=nn.Linear(256,len(state['classes'])).to('cuda:0');head.load_state_dict(stored['state_dict'])
    train,val=at.split_from_report(state['groups'],state['y'],job['split_file'])
    audit=classifier_audit(adapter,head,state['x'],state['groups'],state['y'],state['classes'],state['tags'],train,val,'cuda:0')
    audit.update(coefficient=job['coefficient'],adapter_sha256=ab.digest(Path(job['weights'])/'adapter.pt'),
        classifier_sha256=ab.digest(Path(job['weights'])/'classifier.pt'),batches=state['batches'],
        mean_original_ce=state['ce_sum']/state['batches'],mean_absent_loss=state['absent_sum']/state['batches'],
        mean_training_absent_mass=state['train_absent_mass_sum']/state['batches'])
    if state['batches']!=5000:raise ValueError('Changed training step count')
    ab.write(Path(job['attempt'])/'supervision_audit.json',audit)
    if job['coefficient']==0:
        reference=torch.load(Path(job['historical_weights'])/'adapter.pt',map_location='cpu',weights_only=True)
        differences={k:float((v-reference['state_dict'][k]).abs().max()) for k,v in payload['state_dict'].items()}
        ab.write(Path(job['attempt'])/'control_equivalence.json',dict(max_abs=max(differences.values()),
            bitwise_equal=all(v==0 for v in differences.values()),best_epoch=payload['epoch'],historical_epoch=reference['epoch'],
            note='Fresh original-recipe retraining; no historical classifier checkpoint existed.'))


def worker(job):
    if job['stage']=='train':return train_worker(job)
    if job['stage']=='audit':
        from src.adapter_training import load_training_bank
        x,y,groups,classes,_,_=load_training_bank(job['raw'],job['train_list'])
        mapping=ab.read(Path(job['raw'])/'meta/class_mapping.json')['id_to_dirname']
        tags=parse_tags(ab.read(job['imagelevel_json']),job['train_ids'],mapping)
        ab.write(Path(job['attempt'])/'coverage.json',coverage(groups,y,classes,tags));return
    queue.KEY_IMAGES=[]
    return old.previous.worker(job)


def summarize(results):
    groups={}
    for method in METHODS:
        rows={str(s):results[f'{method}_{s}']['miou'] for s in old.SEEDS if f'{method}_{s}' in results}
        complete=len(rows)==3
        groups[method]=dict(complete=complete,seeds=rows,mean=statistics.mean(rows.values()) if complete else None,
            std=statistics.stdev(rows.values()) if complete else None)
    contrasts={}
    for name,a,b in [('absent-control','absent01','control'),('control-historical','control','historical')]:
        if groups[a]['complete'] and groups[b]['complete']:
            delta={str(s):100*(groups[a]['seeds'][str(s)]-groups[b]['seeds'][str(s)]) for s in old.SEEDS}
            contrasts[name]=dict(by_seed_pp=delta,mean_pp=statistics.mean(delta.values()))
    return dict(groups=groups,contrasts=contrasts,results=results,
        note='VOC val development experiment; unchanged hard bank labels and SupCon; no MIL/cross-fit/soft retrieval/HCR. GT evaluation only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',default='tag_constraint_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--worker',type=Path);p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    previous=ab.read(HISTORY/'manifest.json')
    if ab.read(HISTORY/'status.json')['status']!='complete':raise ValueError('Previous experiment incomplete')
    gmp.verify(previous['code'],'Prior code');gmp.verify(previous['references'],'Prior inputs')
    records=previous['references']+ab.identities([HISTORY/'manifest.json',HISTORY/'status.json'])
    prepared=ab.read(provenance.DEPLOY/'prepared.json');train_jobs={}
    for seed in old.SEEDS:
        file=TRAIN_HISTORY/f'train_{seed}/attempt_001/job.json'
        train_jobs[seed]=ab.read(file);job=train_jobs[seed]
        records+=ab.identities([file,job['split_file'],job['imagelevel_json'],Path(job['weights'])/'training_report.json'])
        if job['raw_signature']!=prepared['jobs'][f'baseline_{seed}']['job']['raw_signature']:raise ValueError('Raw source mismatch')
        mapping=ab.read(Path(job['raw'])/'meta/class_mapping.json')['id_to_dirname']
        parse_tags(ab.read(job['imagelevel_json']),job['train_ids'],mapping)
    identity=old.canonical(dict(protocol='tag_constraint_v1',code=ab.source_identity(),records=records,seeds=list(old.SEEDS),
        methods=METHODS,coefficient=COEFFICIENT,loss='CE+0.1*(-log allowed probability mass)+0.5 SupCon',
        selection='unchanged KNN on frozen internal TRAIN validation images',scope='hard GMP2 library; clean proposals; no HCR; no GT in supervision'))
    output=ROOT/'experiments'/args.version
    if output.exists() and (not args.resume or ab.read(output/'manifest.json')!=identity):raise ValueError('Existing version differs or needs resume')
    queue.ensure_resources(float('inf'))
    members=sum(len(v) for v in ab.bank_members(train_jobs[42]['raw']).values())
    estimate=members*256*4*6+256*2**20
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+estimate:raise ValueError('Need new-bank space plus 5 GiB reserve')
    print(f'[Preflight passed] region coverage audit, 3 historical replays, 6 trainings + 6 evaluations; estimated allowance {estimate/2**30:.2f} GiB',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    stages=ab.Stages(output,True);status='running';error=None;results={};cancel=threading.Event();guard=threading.Lock();start=time.time();deadline=start+24*3600
    entry=queue.__file__;queue.__file__=__file__
    saved={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Experiment interrupted')
    for s in saved:signal.signal(s,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),expected_evaluations=9))
        ab.write(output/'summary.json',dict(status=status,error=error,**summarize(results)))
    def launch_stage(key,job,gpu):
        def action(attempt):
            job.update(attempt=str(attempt),deadline=deadline,metrics=str(attempt/'metrics.json'))
            if job['stage']=='train':
                suffix=Path(args.version)/job['method']/str(job['seed'])/attempt.name
                job['weights']=str(ROOT/'weights/sam3_adapter'/suffix);job['bank']=str(ROOT/'storage/mlp_banks/voc'/suffix)
            env=os.environ.copy();env.update(prepared['jobs'][f'baseline_{job["seed"]}']['command']['environment']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=ROOT/'logs'/args.version/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            result=dict(attempt=str(attempt));artifacts=[attempt,log]
            if job['stage']=='train':
                result.update(weights=job['weights'],bank=job['bank']);artifacts += [Path(job['weights']),Path(job['bank'])]
                if ab.bank_members(job['bank'])!=ab.bank_members(job['raw']):raise ValueError('Changed exported members')
                report=ab.read(Path(job['weights'])/'training_report.json');reference=ab.read(Path(job['historical_weights'])/'training_report.json')
                if report['train_images']!=reference['train_images'] or report['validation_images']!=reference['validation_images']:raise ValueError('Internal split changed')
            return result,artifacts
        return stages.run(key,action)
    def evaluate(method,seed,gpu,trained=None):
        source=prepared['jobs'][f'baseline_{seed}'];job=copy.deepcopy(source['job'])
        if trained:job.update(weights=trained['weights'],bank=trained['bank'])
        value=launch_stage(f'eval_{method}_{seed}',job,gpu);attempt=Path(value['attempt']);metrics=ab.read(attempt/'metrics.json');ref=ab.read(source['metrics'])
        old.validate_eval(evidence.checked_copy(metrics)[0],evidence.checked_copy(ref)[0],job['val_ids'],ref['proposal_count_after'],.2)
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metrics)
        if method=='historical':
            old.require_image_replay(source['reference']['per_image'],str(attempt/'per_image_confusion.npz'))
            if metrics['confusion_matrix']!=ref['confusion_matrix']:raise ValueError('Historical replay mismatch')
        result=dict(miou=evidence.checked_copy(metrics)[0]['mIoU'],metrics=str(attempt/'metrics.json'),per_image=str(attempt/'per_image_confusion.npz'))
        if trained:
            result.update(weights=trained['weights'],bank=trained['bank'],supervision_audit=ab.read(Path(trained['attempt'])/'supervision_audit.json'))
            if method=='control':result['equivalence']=ab.read(Path(trained['attempt'])/'control_equivalence.json')
        with guard:results[f'{method}_{seed}']=result;flush()
    def execute(item,gpu):
        method,seed=item;job=copy.deepcopy(train_jobs[seed]);job.update(method=method,coefficient=0. if method=='control' else COEFFICIENT,historical_weights=job['weights'])
        try:
            trained=launch_stage(f'train_{method}_{seed}',job,gpu);evaluate(method,seed,gpu,trained)
        except BaseException:traceback.print_exc();raise
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        job=copy.deepcopy(train_jobs[42]);job['stage']='audit'
        launch_stage('coverage_audit',job,1)
        queue.run_round(list(old.SEEDS),(1,0),lambda seed,gpu:evaluate('historical',seed,gpu),cancel,deadline,fatal)
        queue.run_round([('control',s) for s in old.SEEDS],(1,0),execute,cancel,deadline,fatal)
        # A gross retraining failure is a pipeline failure, not evidence about tags.
        if abs(statistics.mean(results[f'control_{s}']['miou']-results[f'historical_{s}']['miou'] for s in old.SEEDS))>.001:
            raise ValueError('Fresh control mean differs by more than 0.1 percentage point; inspect control equivalence before absent experiment')
        queue.run_round([('absent01',s) for s in old.SEEDS],(1,0),execute,cancel,deadline,fatal)
        gmp.verify(records,'Inputs');gmp.verify(identity['code'],'Code');status='complete'
    except BaseException as exc:status='failed';error=str(exc);cancel.set();traceback.print_exc();raise
    finally:
        flush();ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(ROOT/'logs'/args.version),
            str(ROOT/'weights/sam3_adapter'/args.version),str(ROOT/'storage/mlp_banks/voc'/args.version)],history_read_only=True))
        queue.__file__=entry
        for s,h in saved.items():signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
