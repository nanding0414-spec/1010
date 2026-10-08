#!/usr/bin/env python3
"""Paired SMFA/GMP MLP seeds; reuse fixed raw banks and the seed-42 results."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_pmfa_ab as ab
from tools import experimental_rpg_ab as rpg
from tools import experimental_gmp_ab as gmp


def training_argv(argv,seed):
    args=list(argv)
    if args.count('--seed')!=1 or '--split-file' not in args:
        raise ValueError('Expected explicit seed and fixed split-file')
    index=args.index('--seed')+1
    if args[index]!='42':raise ValueError('Underlying training recipe changed')
    args[index]=str(seed)
    return args


def image_confusions(predictions,truths,count,ignore):
    import numpy as np
    if len(predictions)!=len(truths):raise ValueError('Prediction/GT counts differ')
    matrices=[]
    for pred,gt in zip(predictions,truths):
        pred=np.asarray(pred);gt=np.asarray(gt)
        if pred.shape!=gt.shape:raise ValueError('Prediction/GT shapes differ')
        valid=(gt!=ignore)&(gt>=0)&(gt<count)
        if ((pred[valid]<0)|(pred[valid]>=count)).any():raise ValueError('Prediction class out of range')
        matrices.append(np.bincount(gt[valid].astype(np.int64)*count+pred[valid].astype(np.int64),
                                   minlength=count*count).reshape(count,count))
    return np.asarray(matrices,dtype=np.int64)


def worker(job):
    if job['stage'] not in ('train','eval'):raise ValueError('Seed experiment only trains/evaluates; no extraction')
    if job['stage']=='train':
        original=ab.runpy.run_path
        def run(path,*args,**kwargs):
            if Path(path).resolve()!=ROOT/'tools/train_feature_adapter.py':raise ValueError('Unexpected training entry')
            sys.argv=training_argv(sys.argv,job['seed'])
            ab.write(Path(job['attempt'])/'effective_training_argv.json',dict(argv=sys.argv,seed=job['seed']))
            return original(path,*args,**kwargs)
        ab.runpy.run_path=run
    else:
        import numpy as np
        import project_utils.metrics as metrics
        original=metrics.evaluate_dataset_miou
        def evaluate(preds,gts,count,ignore,*args,**kwargs):
            if len(preds)!=len(job['val_ids']):raise ValueError('Incomplete per-image evaluation')
            matrices=image_confusions(preds,gts,count,ignore)
            np.savez_compressed(Path(job['attempt'])/'per_image_confusion.npz',
                                ids=np.asarray(job['val_ids'],dtype=str),confusions=matrices)
            return original(preds,gts,count,ignore,*args,**kwargs)
        metrics.evaluate_dataset_miou=evaluate
    if job['mode']=='gmp':gmp.worker(job)
    elif job['mode']=='smfa':ab.worker(job)
    else:raise ValueError('Unsupported mode')


def previous(folder):
    manifest=ab.read(folder/'manifest.json');comparison=ab.read(folder/'comparison.json')
    selected=ab.read(folder/'selection.json')
    if ab.read(folder/'status.json')['status']!='complete' or comparison['status']!='complete':
        raise ValueError('Previous GMP run must be complete')
    if (selected['winner']!='rho01' or comparison['selected']!='rho01' or comparison['rho']!=.1
        or selected['final_val_used'] is not False or comparison['fixed_members'] is not True):
        raise ValueError('This protocol locks the internally selected rho=0.1 fixed-member experiment')
    gmp.verify(manifest['code'],'Previous GMP source');gmp.verify(manifest['inputs'],'Previous GMP inputs')
    rpg.verify_records=gmp.verify
    old,smfa,records=rpg.baseline(Path(manifest['previous']))
    current={}
    for stage in ('merge_rho01','train_rho01','final_eval'):
        marker=ab.read(folder/stage/'complete.json');gmp.verify(marker['artifacts'],stage)
        records+=marker['artifacts']
        current.update({k:v for k,v in marker['result'].items() if k!='worker_report'})
    expected=gmp.signature({},'gmp',gmp.GMP(.1))['experimental_pooling']
    sig=ab.read(Path(current['raw'])/'meta/feature_signature.json')
    if sig['format_version']!=3004 or sig['experimental_pooling']!=expected:
        raise ValueError('GMP raw bank signature differs')
    if ab.bank_members(smfa['raw'])!=ab.bank_members(current['raw']):raise ValueError('Raw bank members differ')
    split=ab.read(Path(smfa['weights'])/'training_report.json')
    other=ab.read(Path(current['weights'])/'training_report.json')
    if any(split[k]!=other[k] for k in ('train_images','validation_images')):raise ValueError('Seed42 split differs')
    records+=ab.identities([folder/name for name in ('manifest.json','selection.json','comparison.json','status.json')])
    return old,smfa,current,records


def validate_image_matrices(path,metrics):
    import numpy as np
    with np.load(path,allow_pickle=False) as data:
        if data['ids'].tolist()!=metrics['evaluated_ids']:raise ValueError('Per-image IDs disagree')
        m=data['confusions']
        if m.shape!=(len(data['ids']),21,21) or (m<0).any():raise ValueError('Invalid per-image confusion')
        if not np.array_equal(m.sum(0),np.asarray(metrics['confusion_matrix'])):raise ValueError('Per-image sum disagrees')


def summary(reports):
    rows=[dict(seed=int(seed),smfa_miou=r['smfa_miou'],gmp_miou=r['gmp_miou'],delta_pp=r['delta_pp'])
          for seed,r in sorted(reports.items())]
    deltas=[r['delta_pp'] for r in rows]
    if not all(math.isfinite(x) for x in deltas):raise ValueError('Nonfinite seed result')
    return dict(status='complete',rho=.1,results=rows,mean_delta_pp=statistics.mean(deltas),
        sample_std_delta_pp=statistics.stdev(deltas) if len(deltas)>1 else None,
        minimum_delta_pp=min(deltas),positive_seeds=sum(d>0 for d in deltas),
        note='Seed42 reused; fixed internal image split; no parameter reselection or significance claim.',
        per_class_delta_pp={c:{str(seed):r['per_class_iou'][c]['delta_pp'] for seed,r in reports.items()}
                            for c in next(iter(reports.values()))['per_class_iou']})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--previous',type=Path,help='Complete GMP experiment directory; auto-select only if unique')
    parser.add_argument('--version',default='gmp_seeds_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    opts=parser.parse_args(argv);os.chdir(ROOT)
    if opts.worker:return worker(ab.read(opts.worker))
    ab.parse_args(['--version',opts.version])
    if min(opts.gpus)<0 or len(set(opts.gpus))!=2:parser.error('Two distinct physical GPU IDs required')
    if opts.previous is None:
        candidates=[]
        for file in (ROOT/'experiments').glob('*/comparison.json'):
            data=ab.read(file)
            if data.get('status')=='complete' and data.get('selected')=='rho01' and data.get('fixed_members'):
                candidates.append(file.parent)
        if len(candidates)!=1:parser.error('Specify --previous; complete GMP candidates: '+str(candidates))
        opts.previous=candidates[0]
    opts.previous=opts.previous.resolve()
    old,smfa,selected,records=previous(opts.previous)
    flags=['--version',opts.version,'--gpu',str(opts.gpus[0])]
    for key in ('dataset_root','image_dir','checkpoint','train_list','val_list','imagelevel_json','gt_dir'):
        flags+=['--'+key.replace('_','-'),old['options'][key]]
    baseline_folder=Path(ab.read(opts.previous/'manifest.json')['previous'])
    for kind,flag in (('pseudo','--pseudo-dir'),('proposals','--proposals-dir')):
        directory=old['selected'].get(kind) or ab.read(baseline_folder/kind/'complete.json')['result'][kind]
        flags += [flag,directory]
    args=ab.parse_args(flags);context=ab.preflight(args);ab.check_gpu(opts.gpus[1])
    if not all(context['selected'].values()):raise ValueError('Shared inputs incomplete; will not regenerate')
    reference_report=gmp.report(smfa,selected,context['val_ids'])
    output=ROOT/'experiments'/opts.version;logs=ROOT/'logs'/opts.version
    weights=ROOT/'weights/sam3_adapter'/opts.version;banks=ROOT/'storage/mlp_banks/voc'/opts.version
    identity=dict(previous=str(opts.previous),gpus=opts.gpus,seeds=[123,2026],rho=.1,
        inputs=context['inputs'],code=context['code'],reference_artifacts=records,protocol='paired_fixed_split_v1')
    if output.exists():
        if not opts.resume or ab.read(output/'manifest.json')!=identity:raise ValueError('Version identity differs or --resume missing')
    elif any(p.exists() for p in (logs,weights,banks)):raise FileExistsError('Version already used')
    print(f'[Plan] previous={opts.previous}; seed42 reused; GPU{opts.gpus[0]}=123, GPU{opts.gpus[1]}=2026; train/eval only',flush=True)
    if opts.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    cancel=threading.Event();old_signal=signal.getsignal(signal.SIGTERM)
    def stop(*_):cancel.set();raise KeyboardInterrupt('Seed experiment stopped')
    signal.signal(signal.SIGTERM,stop);ab.SCRIPT=Path(__file__).resolve();stages=ab.Stages(output,opts.resume)
    common={k:str(getattr(args,k)) for k in ('dataset_root','image_dir','imagelevel_json','checkpoint','train_list','val_list','gt_dir')}
    common.update(train_ids=context['train_ids'],val_ids=context['val_ids'],images=context['images'])
    reports={42:reference_report}
    from configs.experiment import CLEANUP
    def execute(kind,mode,seed,gpu,result):
        key=f'{mode}_seed{seed}_{kind}'
        if cancel.is_set():raise RuntimeError('Cancelled')
        def action(attempt):
            job=dict(common,stage=kind,mode=mode,seed=seed,attempt=str(attempt),
                     pmfa=asdict(gmp.GMP(.1)) if mode=='gmp' else asdict(ab.PMFA()))
            env=ab.base_env(gpu);env['OVERRIDE_SAM3_CKPT']=str(args.checkpoint)
            artifacts=[attempt];values={}
            if kind=='train':
                w=weights/f'{mode}_seed{seed}'/attempt.name;b=banks/f'{mode}_seed{seed}'/attempt.name
                job.update(raw=result['raw'],weights=str(w),bank=str(b),
                           split_file=str(Path(smfa['weights'])/'training_report.json'))
                artifacts += [w,b];values.update(weights=str(w),bank=str(b))
            else:
                job.update(metrics=str(attempt/'metrics.json'),cleanup=str(output/'cleanup.json'))
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(result['weights'])/'adapter.pt'),
                           OVERRIDE_INF_FEATURE_BANK_ROOT=result['bank'],OVERRIDE_INF_PROPOSALS_DIR=context['selected']['proposals'])
                values.update(metrics=job['metrics'],per_image=str(attempt/'per_image_confusion.npz'))
            log=logs/f'{key}_{attempt.name}.log';ab.launch(job,attempt,env,log,cancel=cancel)
            if ab.read(attempt/'worker_report.json')['status']!='complete':raise ValueError('Worker incomplete')
            if kind=='train':
                report=ab.read(Path(values['weights'])/'training_report.json')
                split=ab.read(Path(smfa['weights'])/'training_report.json')
                if any(report[k]!=split[k] for k in ('train_images','validation_images')):raise ValueError('Internal split changed')
                effective=ab.read(attempt/'effective_training_argv.json')['argv']
                if effective[effective.index('--seed')+1]!=str(seed):raise ValueError('Effective seed mismatch')
            else:validate_image_matrices(values['per_image'],ab.read(values['metrics']))
            values['worker_report']=str(attempt/'worker_report.json')
            return values,artifacts+[log]
        return stages.run(key,action)
    def lane(seed,gpu):
        results={}
        for mode,source in (('smfa',smfa),('gmp',selected)):
            results[mode]=dict(raw=source['raw'])
            for kind in ('train','eval'):
                results[mode].update(execute(kind,mode,seed,gpu,results[mode]))
        value=gmp.report(results['smfa'],results['gmp'],context['val_ids'])
        value.update(seed=seed,rho=.1,per_image_files={m:r['per_image'] for m,r in results.items()})
        reports[seed]=value;ab.write(output/f'comparison_seed{seed}.json',value)
    try:
        ab.write(output/'cleanup.json',CLEANUP)
        ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,logs,weights,banks])),
                 read_only_raw_banks=[smfa['raw'],selected['raw']],read_only_reference=str(opts.previous),automatic_deletion=False))
        ab.write(output/'comparison_seed42.json',dict(reference_report,seed=42,rho=.1,reused=True,
                 per_image_note='Historical seed42 has no per-image confusion export; no re-evaluation requested.'))
        gmp.parallel([lambda:lane(123,opts.gpus[0]),lambda:lane(2026,opts.gpus[1])],cancel)
        stages.verify_all();gmp.verify(records,'Reference artifacts');gmp.verify(context['inputs'],'Inputs')
        if ab.source_identity()!=context['code']:raise ValueError('Source changed during experiment')
        value=summary(reports)
        value['workers']={p.parent.name:ab.read(ab.read(p)['result']['worker_report']) for p in output.glob('*/complete.json')}
        ab.write(output/'summary.json',value)
        (output/'summary.csv').write_text('seed,smfa_miou,gmp_miou,delta_pp\n'+''.join(
            f'{r["seed"]},{r["smfa_miou"]},{r["gmp_miou"]},{r["delta_pp"]}\n' for r in value['results']),encoding='utf-8')
        ab.write(output/'status.json',dict(status='complete'))
        print(f'[DONE] paired mean delta={value["mean_delta_pp"]:+.5f} pp; {output}/summary.json',flush=True)
    except BaseException as exc:
        cancel.set();ab.write(output/'status.json',dict(status='incomplete',error=str(exc),completed_seeds=sorted(reports)));raise
    finally:
        signal.signal(signal.SIGTERM,old_signal);lock.unlink()


if __name__=='__main__':main()
