"""Count-matched background representation experiment, fixed GMP2, three seeds.

Training GT is never read. Stream new train-tag Active masks; retain only chosen
background masks and vectors. Production and earlier experiment files stay frozen.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import experimental_tag_constraint_queue as tag
from tools.experimental_tag_constraint_recovery import evaluation_environment,stages_with_recovery
ab,queue,gmp,old=tag.ab,tag.queue,tag.gmp,tag.old
PROTOCOL='background_partition_v1'
GROUPS=('merged_matched','split_matched','safe_matched')
ACTIVE=dict(preview=False,bucket_size=4,neighbor_tokens=2,hypernym_tokens=2,
    context_strength=1.,confidence_threshold=.15,mask_nms_threshold=.7,
    min_mask_area=100,min_mask_region_area=0,max_mask_area_ratio=.95,max_masks=0)


def binary(mask):
    m=np.asarray(mask)
    if m.ndim!=2 or min(m.shape)<1 or not np.isfinite(m).all() or not np.isin(m,[0,1]).all():
        raise ValueError('Expected finite binary mask')
    return m.astype(bool)


def extraction_context(train, reference):
    """Historical train jobs intentionally omitted extraction-only fields."""
    for key in ('cache','checkpoint_sha256','train_ids','raw_signature'):
        if train[key]!=reference[key]:raise ValueError('Historical extraction/training mismatch: '+key)
    result=copy.deepcopy(train)
    for key in ('plan','pseudo','source'):
        if key not in reference:raise ValueError('Missing verified extraction field: '+key)
        result[key]=copy.deepcopy(reference[key])
    return result


def unknown_union(background, proposals):
    bg=binary(background);unknown=np.zeros_like(bg);accepted=[]
    for i,proposal in enumerate(proposals):
        m=binary(proposal)
        if m.shape!=bg.shape:raise ValueError('Proposal/background geometry differs')
        ys,xs=np.nonzero(m)
        if not len(xs):continue
        cy,cx=int(np.floor(ys.mean()+.5)),int(np.floor(xs.mean()+.5))
        cy,cx=min(cy,bg.shape[0]-1),min(cx,bg.shape[1]-1)
        overlap=float((m&bg).sum()/m.sum())
        if bg[cy,cx] and overlap>.5:
            unknown |= m&bg;accepted.append(dict(index=i,background_fraction=overlap))
    return unknown,accepted


def local_mask(background, side, stem, min_area=300):
    bg=binary(background)
    if type(side) is not int or not 1<=side<=72:raise ValueError('Invalid tile side in tokens')
    h,w=bg.shape;boxes=[]
    for ty in range(0,72,side):
        for tx in range(0,72,side):
            y0,y1=ty*h//72,min(ty+side,72)*h//72
            x0,x1=tx*w//72,min(tx+side,72)*w//72
            if y1>y0 and x1>x0:
                rank=hashlib.sha256(f'{stem}:{ty}:{tx}'.encode()).hexdigest()
                boxes.append((rank,(y0,y1,x0,x1)))
    for _,(y0,y1,x0,x1) in sorted(boxes):
        if int(bg[y0:y1,x0:x1].sum())>=min_area:
            result=np.zeros_like(bg);result[y0:y1,x0:x1]=bg[y0:y1,x0:x1]
            return result,[y0,y1,x0,x1]
    return np.zeros_like(bg),None


def bank_signature(base, method, side):
    if method not in GROUPS:raise ValueError('Unknown background method')
    return dict(base,background_protocol=dict(format=PROTOCOL,method=method,tile_side_tokens=side,
        tile_rule='ceil(sqrt(median retained foreground hard-center support))',
        tile_choice='SHA256(image:tile-row:tile-column), first area>=300',
        unknown='proposal centroid in pseudo-background and overlap strictly >0.5',
        active=ACTIVE,sampling='one background per eligible image, same images across three groups',
        foreground='original filtered members and vectors, no new filter',query='unchanged cached GMP C'))


def check_geometry(job, stem):
    from src.region_extractors import get_regions_from_pseudo
    identity=job['cache_identity'][stem]
    if ab.digest(job['images'][stem])!=identity['image_sha256'] or ab.digest(Path(job['pseudo'])/(stem+'.png'))!=identity['mask_source_sha256']:
        raise ValueError('Training image/pseudo changed: '+stem)
    regions=get_regions_from_pseudo(job['images'][stem],300,job['pseudo'])
    rows=ab.read(job['plan'])[stem] if '_plan' not in job else job['_plan'][stem]
    if queue.mask_hashes([regions[r['index']]['mask'] for r in rows])!=[r['mask_sha256'] for r in rows]:
        raise ValueError('Original member mask changed')
    return regions,rows


def geometry_worker(job):
    supports=[];backgrounds=0;job['_plan']=ab.read(job['plan'])
    for n,stem in enumerate(job['train_ids'],1):
        queue.ensure_resources(job['deadline']);regions,rows=check_geometry(job,stem)
        for row in rows:
            if row['cls']=='background':backgrounds+=1
            else:supports.append(queue.mpsa.selection(regions[row['index']]['mask'],(0,0))[1]['support'])
        if n%1000==0:print(f'[Geometry] {n}/{len(job["train_ids"])}',flush=True)
    median=float(np.median(supports));side=max(1,min(72,math.ceil(math.sqrt(median))))
    ab.write(Path(job['attempt'])/'geometry.json',dict(foreground_regions=len(supports),background_regions=backgrounds,
        median_foreground_support=median,tile_side_tokens=side,source='TRAIN masks, no GT'))


def pool(model, tokens, mask):
    import torch
    import torch.nn.functional as F
    indices,info=queue.mpsa.selection(mask,(0,0))
    if not len(indices):raise ValueError('Empty selected background')
    selected=tokens[0,torch.as_tensor(indices,device=tokens.device)].float()
    value,solve=gmp.solve(selected,torch.ones(len(selected),device=tokens.device),.1)
    value=F.normalize(torch.stack([value]).mean(0),dim=0).cpu().numpy()
    if not np.isfinite(value).all() or np.linalg.norm(value)<1e-6:raise ValueError('Invalid background GMP vector')
    return value,dict(**info,solve=solve)


def extract_worker(job):
    import torch
    from PIL import Image
    from src.feature_extractor_factory import build_feature_extractor
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    from dataclasses import asdict
    model=build_feature_extractor(torch.device('cuda:0'),use_interactivity=False)
    if model.feature_signature()!=job['source']:raise ValueError('Live token feature signature changed')
    vocabulary=SAM3PseudoMaskGenerator(dataset_type='voc',load_model=False,output_dir=job['cache_output'])
    if old.canonical([asdict(c) for c in vocabulary.active_classes])!=old.canonical(job['active_classes']):raise ValueError('Active vocabulary changed')
    proposer=SAM3ActiveProposer(model.model,torch.device('cuda:0'),vocabulary.active_classes,**ACTIVE)
    labels={str(r['img_id']):r['labels'] for r in ab.read(job['imagelevel_json'])['images']}
    job['_plan']=ab.read(job['plan'])
    for n,stem in enumerate(job['ids'],1):
        queue.ensure_resources(job['deadline']);folder=Path(job['cache_output'])/stem
        identity=dict(input=job['cache_identity'][stem],protocol=PROTOCOL,side=job['side'],active=ACTIVE)
        committed=queue.cache_read(folder,identity)
        if committed:continue
        start=time.monotonic();regions,rows=check_geometry(job,stem)
        bg_rows=[r for r in rows if r['cls']=='background']
        if len(bg_rows)!=1:raise ValueError('Expected one original background per train image')
        row=bg_rows[0];bg=binary(regions[row['index']]['mask'])
        with Image.open(job['images'][stem]) as im:rgb=np.asarray(im.convert('RGB'))
        permitted=list(dict.fromkeys(int(v)-1 for v in labels[stem] if int(v)!=0))
        with torch.inference_mode():output=proposer(np.ascontiguousarray(rgb[...,::-1]),class_indices=permitted)
        proposals=output['instances'].pred_masks.numpy()
        unknown,selected=unknown_union(bg,proposals)
        split,box=local_mask(bg,job['side'],stem)
        safe,safe_box=local_mask(bg&~unknown,job['side'],stem)
        eligible=box is not None and safe_box is not None
        old_vectors=queue.cache_vectors(Path(job['cache'])/'train'/stem,job['cache_identity'][stem],'C',[r['mask_sha256'] for r in rows])
        original=old_vectors[rows.index(row)]
        diagnostics={};vectors=np.zeros((3,1280),np.float32);vectors[0]=original
        if eligible or job['stage']=='probe':
            with torch.inference_mode():
                inp,_=model._preprocess_image(rgb);tokens,grid=model._forward_tokens(inp)
                if tuple(grid)!=(72,72) or tokens.shape!=(1,5184,1280):raise ValueError('Changed token geometry')
                if eligible:
                    vectors[1],diagnostics['split']=pool(model,tokens,split)
                    vectors[2],diagnostics['safe']=pool(model,tokens,safe)
                if job['stage']=='probe':
                    reproduced,diagnostics['merged']=pool(model,tokens,bg)
                    error=float(np.max(np.abs(reproduced-original)))
                    if not np.allclose(reproduced,original,atol=5e-5,rtol=5e-5):raise ValueError('Original GMP background not reproduced')
                    diagnostics['original_max_abs']=error
                    if eligible:
                        same,_=pool(model,tokens,split)
                        if not np.array_equal(same,vectors[1]):raise ValueError('Repeated shared token aggregation differs')
                del tokens
        attempt=queue.next_attempt(folder);file=attempt/'background.npz'
        np.savez_compressed(file,vectors=vectors,split=np.packbits(split),safe=np.packbits(safe),shape=np.asarray(bg.shape))
        report=dict(image=stem,sample_id=row['id'],eligible=eligible,original_mask_sha256=row['mask_sha256'],
            mask_hashes=queue.mask_hashes([split,safe]),split_box=box,safe_box=safe_box,
            original_area=int(bg.sum()),split_area=int(split.sum()),safe_area=int(safe.sum()),
            unknown_pixels=int(unknown.sum()),proposal_count=len(proposals),unknown_candidates=selected,
            allowed_class_ids=[v+1 for v in permitted],proposal_source='new TRAIN-tag Active, not historical replay',
            diagnostics=diagnostics,seconds=time.monotonic()-start)
        ab.write(attempt/'regions.json',report)
        ab.write(folder/'complete.json',dict(identity=identity,file=str(file),report=str(attempt/'regions.json'),artifacts=ab.identities([file,attempt/'regions.json'])))
        del proposals,output;proposer.last_result=None
        print(f'[Background] {n}/{len(job["ids"])} {stem} eligible={eligible} unknown={report["unknown_pixels"]}',flush=True)


def worker(job):
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'sam3-main'))
    import torch
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();started=time.monotonic();status='failed';error=None;category='local'
    try:
        if job['stage']=='diagnostic':diagnostic_worker(job)
        elif job['stage'] in ('geometry','probe','extract'):
            ab.install_precision()
            from project_utils.seed_utils import set_global_seed
            set_global_seed(42)
            if job['stage']=='geometry':geometry_worker(job)
            else:extract_worker(job)
        else:
            if job.get('background_method'):
                import src.feature_signature as fs
                fs.REQUIRED_SIGNATURE_KEYS.add('background_protocol')
                if job['raw_signature']!=bank_signature(job['base_signature'],job['background_method'],job['side']):raise ValueError('Background source protocol differs')
            queue.KEY_IMAGES=[]
            old.previous.worker(job)
        torch.cuda.synchronize();status='complete'
    except queue.Deferred as exc:category='deferred';error=str(exc);raise
    except torch.cuda.OutOfMemoryError as exc:category='oom';error=str(exc);raise
    except (ValueError,KeyError,FileNotFoundError,queue.GlobalFailure) as exc:category='global';error=str(exc);raise
    except BaseException as exc:error=str(exc);raise
    finally:ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,category=category,error=error,stage=job['stage'],seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),precision_policy=ab.POLICY))


def merge_banks(job,cache,raw_root,side,attempt):
    from src.feature_bank_storage import load_feature_matrix_npz,save_feature_matrix_npz
    original_files=list((Path(job['raw'])/'features/background').glob('*.npz'))
    if len(original_files)!=1:raise ValueError('Ambiguous background bank')
    bg,ids=load_feature_matrix_npz(original_files[0]);lookup={str(s):i for i,s in enumerate(ids)}
    matrices={m:[] for m in GROUPS};kept=[];records=[];excluded=[];stats=[]
    for stem in job['train_ids']:
        identity=dict(input=job['cache_identity'][stem],protocol=PROTOCOL,side=side,active=ACTIVE)
        value=queue.cache_read(cache/stem,identity)
        if value is None:raise ValueError('Missing committed background cache: '+stem)
        d=ab.read(value['report']);records+=value['artifacts'];stats.append(d)
        with np.load(value['file'],allow_pickle=False) as data:v=data['vectors'].copy()
        if not d['eligible']:excluded.append(stem);continue
        if not np.allclose(v[0],bg[lookup[d['sample_id']]],atol=1e-7,rtol=1e-7):raise ValueError('Original background vector changed')
        if not np.isfinite(v).all() or (np.linalg.norm(v,axis=1)<1e-6).any():raise ValueError('Invalid retained background')
        kept.append(d['sample_id'])
        for i,m in enumerate(GROUPS):matrices[m].append(v[i])
    if not kept:raise ValueError('No safe background remains')
    paths={}
    for m in GROUPS:
        root=raw_root/m/attempt.name;root.mkdir(parents=True,exist_ok=False)
        for source in (Path(job['raw'])/'features').glob('*/*.npz'):
            if source.parent.name=='background':continue
            target=root/'features'/source.parent.name/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(source.resolve())
        save_feature_matrix_npz(root/'features/background/features.npz',np.stack(matrices[m]),kept)
        shutil.copy2(Path(job['raw'])/'meta/class_mapping.json',root/'class_mapping.tmp')
        ab.write(root/'meta/feature_signature.json',bank_signature(job['raw_signature'],m,side))
        os.replace(root/'class_mapping.tmp',root/'meta/class_mapping.json');paths[m]=str(root)
        if ab.bank_members(root)['background']!=set(kept):raise ValueError('Background members differ')
        for cls,member in ab.bank_members(job['raw']).items():
            if cls!='background' and ab.bank_members(root)[cls]!=member:raise ValueError('Foreground members differ')
    ab.write(attempt/'background_audit.json',dict(original_background_count=len(ids),matched_count=len(kept),excluded_images=excluded,
        unknown_candidate_count=sum(len(r['unknown_candidates']) for r in stats),images_with_unknown=sum(r['unknown_pixels']>0 for r in stats),
        mean_unknown_background_fraction=statistics.mean(r['unknown_pixels']/r['original_area'] for r in stats),
        training_gt_read=False,member_count_control='same images/sample IDs across all three new banks',foreground_storage='read-only symlinks'))
    return dict(raw=paths,audit=str(attempt/'background_audit.json')),[attempt,*map(Path,paths.values())]


def diagnostic_worker(job):
    """GT is opened only after all masks/cohort/banks have been frozen."""
    from PIL import Image
    rows={m:[] for m in GROUPS}
    for stem in job['train_ids']:
        queue.ensure_resources(job['deadline'])
        value=queue.cache_read(Path(job['cache_output'])/stem)
        d=ab.read(value['report'])
        if not d['eligible']:continue
        with np.load(value['file'],allow_pickle=False) as data:
            shape=tuple(data['shape']);size=int(np.prod(shape))
            masks=[np.unpackbits(data[k])[:size].reshape(shape).astype(bool) for k in ('split','safe')]
        if queue.mask_hashes(masks)!=d['mask_hashes']:raise ValueError('Diagnostic mask hash mismatch')
        with Image.open(Path(job['pseudo'])/(stem+'.png')) as im:bg=np.asarray(im)==0
        with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im:gt=np.asarray(im)
        for m,mask in zip(GROUPS,[bg,*masks]):
            h=tag.provenance.histogram(mask,gt)
            rows[m].append(dict(image=stem,sample_id=d['sample_id'],**h,
                foreground_fraction=float(sum(h['histogram'][1:])/h['valid_pixels']) if h['valid_pixels'] else None))
    report={}
    for m,data in rows.items():
        valid=[r for r in data if r['foreground_fraction'] is not None]
        report[m]=dict(regions=len(data),valid_regions=len(valid),mean_foreground_fraction=statistics.mean(r['foreground_fraction'] for r in valid) if valid else None,
            regions_foreground_ge_half=sum(r['foreground_fraction']>=.5 for r in valid),
            regions_foreground_ge_tenth=sum(r['foreground_fraction']>=.1 for r in valid))
    ab.write(Path(job['attempt'])/'background_purity.json',dict(summary=report,rows=rows,gt_role='post-construction diagnostic only; no selection/training input'))


def summarize(results):
    groups={};contrasts={}
    for m in ('historical',*GROUPS):
        values={str(s):results[f'{m}_{s}']['miou'] for s in old.SEEDS if f'{m}_{s}' in results}
        complete=len(values)==3
        groups[m]=dict(complete=complete,seeds=values,mean=statistics.mean(values.values()) if complete else None,std=statistics.stdev(values.values()) if complete else None)
    for a,b in [('merged_matched','historical'),('split_matched','merged_matched'),('safe_matched','split_matched'),('safe_matched','merged_matched')]:
        if groups[a]['complete'] and groups[b]['complete']:
            d={str(s):100*(groups[a]['seeds'][str(s)]-groups[b]['seeds'][str(s)]) for s in old.SEEDS}
            contrasts[a+'-'+b]=dict(by_seed_pp=d,mean_pp=statistics.mean(d.values()))
    return dict(groups=groups,contrasts=contrasts,results=results,note='Fixed TRAIN geometry/tags only; VOC val development, no HCR or automatic deployment.')


def verified_recovery(folder,active_classes):
    manifest=ab.read(folder/'manifest.json')
    if manifest['protocol']!=PROTOCOL or old.canonical(manifest['active'])!=old.canonical(ACTIVE) or old.canonical(manifest['active_classes'])!=old.canonical(active_classes):
        raise ValueError('Recovery protocol/vocabulary differs')
    unchanged=[];records=[]
    for r in manifest['code']:
        if Path(r['path']).resolve()==Path(__file__).resolve():
            # Explicit exception only for this independent launcher, whose
            # prior bytes must be preserved exactly; all other hashes retained.
            archived=folder/'source_snapshot/experimental_background_partition_queue.py'
            record=dict(r,path=str(archived));gmp.verify([record],'Archived prior launcher');records.append(record)
        else:unchanged.append(r)
    gmp.verify(unchanged,'Unchanged recovery source');gmp.verify(manifest['records'],'Recovery inputs');ab.Stages(folder,True).verify_all()
    records+=unchanged+manifest['records']+ab.identities([folder/'manifest.json'])
    completed={}
    for key in ['geometry',*[f'eval_historical_{s}' for s in old.SEEDS]]:
        marker=folder/key/'complete.json'
        if marker.exists():
            value=ab.read(marker);completed[key]=value
            records+=ab.identities([marker])+value['artifacts']
    return records,completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--version',default='background_partition_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--previous',type=Path,default=ROOT/'experiments/tag_constraint_fixed_20261005_133142')
    parser.add_argument('--recover-previous',type=Path)
    parser.add_argument('--gpus',nargs=2,type=int,default=[1,0]);parser.add_argument('--max-hours',type=float,default=48)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    opts=parser.parse_args();os.chdir(ROOT)
    if opts.worker:return worker(ab.read(opts.worker))
    ab.parse_args(['--version',opts.version])
    if len(set(opts.gpus))!=2 or min(opts.gpus)<0 or not math.isfinite(opts.max_hours) or opts.max_hours<=0:raise ValueError('Invalid GPUs/time')
    previous=opts.previous.resolve();manifest=ab.read(previous/'manifest.json');summary=ab.read(previous/'summary.json')
    if summary['status']!='complete' or not all(summary['groups'][m]['complete'] for m in ('historical','control','absent01')):raise ValueError('Completed previous experiment required')
    gmp.verify(manifest['code'],'Previous source');gmp.verify(manifest['records'],'Previous inputs');ab.Stages(previous,True).verify_all()
    prepared=ab.read(tag.provenance.DEPLOY/'prepared.json');refs={s:prepared['jobs'][f'baseline_{s}'] for s in old.SEEDS}
    train=extraction_context(ab.read(tag.TRAIN_HISTORY/'train_42/attempt_001/job.json'),refs[42]['job']);train.pop('method',None)
    if not Path(train['plan']).is_file() or not Path(train['pseudo']).is_dir():raise FileNotFoundError('Historical region plan/pseudo directory missing')
    plan=ab.read(train['plan'])
    if set(plan)!=set(train['train_ids']):raise ValueError('Region plan does not cover fixed training images')
    records=manifest['records']+ab.identities([previous/'manifest.json',previous/'summary.json',tag.TRAIN_HISTORY/'train_42/attempt_001/job.json'])
    records+=ab.identities([train['plan']])
    for s in old.SEEDS:records+=ab.identities([refs[s]['metrics'],refs[s]['reference']['per_image']])
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    from dataclasses import asdict
    vocabulary=SAM3PseudoMaskGenerator(dataset_type='voc',load_model=False)
    active_classes=[asdict(c) for c in vocabulary.active_classes]
    recovered={};recovery=None
    if opts.recover_previous:
        folder=opts.recover_previous.resolve();extra,recovered=verified_recovery(folder,active_classes);records+=extra
        recovery=dict(previous=str(folder),reused_stages=sorted(recovered),fix='normalize vocabulary tuple/list serialization only')
    for s in old.SEEDS:
        if summary['results'][f'control_{s}']['equivalence']['max_abs']!=0:raise ValueError('Historical control not exact')
    for gpu in opts.gpus:ab.check_gpu(gpu)
    # New backgrounds + 9 projected banks, per-image packed masks, reports; foreground raw files shared read-only.
    n=len(train['train_ids']);estimate=n*1280*4*5+37200*256*4*9+384*2**20
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+estimate:raise queue.Deferred(f'Need {estimate/2**30:.2f} GiB allowance plus 5 GiB reserve')
    identity=old.canonical(dict(protocol=PROTOCOL,code=ab.source_identity(),records=records,active=ACTIVE,active_classes=active_classes,
        previous=str(previous),recovery=recovery,gpus=opts.gpus,seeds=list(old.SEEDS),groups=GROUPS,background_count='one per common eligible image',no_gt_in_method=True))
    output=ROOT/'experiments'/opts.version;cache=ROOT/'storage/raw_banks/voc'/opts.version/'background_cache';raw_root=cache.parent
    if output.exists() and (not opts.resume or ab.read(output/'manifest.json')!=identity):raise ValueError('Existing version differs or needs resume')
    if not output.exists() and any(p.exists() for p in [raw_root,ROOT/'weights/sam3_adapter'/opts.version,ROOT/'storage/mlp_banks/voc'/opts.version,ROOT/'logs'/opts.version]):raise FileExistsError('Version paths already used')
    print(f'[Preflight passed] 3 historical replays, streamed train-tag proposals + background pass, 9 new trainings/evaluations; allowance {estimate/2**30:.2f} GiB',flush=True)
    if opts.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists():ab.write(budget,dict(start=time.time()))
    started=ab.read(budget)['start'];deadline=started+opts.max_hours*3600;cancel=threading.Event();guard=threading.Lock();status='running';error=None;results={};failures={}
    stage_class=stages_with_recovery(ab.Stages,opts.recover_previous.resolve(),recovered) if recovered else ab.Stages
    stages=stage_class(output,True);oldentry=queue.__file__;queue.__file__=__file__;handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Interrupted')
    for s in handlers:signal.signal(s,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),failures=failures,expected_evaluations=12))
        ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,**summarize(results)))
    def launch(key,job,gpu):
        def action(attempt):
            job.update(attempt=str(attempt),deadline=deadline,metrics=str(attempt/'metrics.json'))
            if job['stage']=='train':
                suffix=Path(opts.version)/job['background_method']/str(job['seed'])/attempt.name
                job.update(weights=str(ROOT/'weights/sam3_adapter'/suffix),bank=str(ROOT/'storage/mlp_banks/voc'/suffix))
            env=os.environ.copy();env.update(refs[job['seed']]['command']['environment']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            if job['stage'] in ('geometry','probe','extract'):env['OVERRIDE_FEATURE_ADAPTER_CKPT']=''
            env=evaluation_environment(job,env);log=ROOT/'logs'/opts.version/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            value=dict(attempt=str(attempt));artifacts=[attempt,log]
            if job['stage'] in ('probe','extract'):
                for stem in job['ids']:
                    marker=cache/stem/'complete.json';data=ab.read(marker)
                    artifacts += [marker,*[Path(r['path']) for r in data['artifacts']]]
            if job['stage']=='train':
                value.update(weights=job['weights'],bank=job['bank']);artifacts += [Path(job['weights']),Path(job['bank'])]
                if ab.bank_members(job['raw'])!=ab.bank_members(job['bank']):raise ValueError('Exported members differ')
                report=ab.read(Path(job['weights'])/'training_report.json');reference=ab.read(Path(train['weights'])/'training_report.json')
                if set(report['validation_images'])-set(reference['validation_images']):raise ValueError('Frozen internal validation changed')
                value['internal_image_coverage']=dict(train_images=len(report['train_images']),validation_images=len(report['validation_images']),
                    missing_original_train=sorted(set(reference['train_images'])-set(report['train_images'])),
                    missing_original_validation=sorted(set(reference['validation_images'])-set(report['validation_images'])))
            return value,artifacts
        return stages.run(key,action)
    def evaluate(m,s,gpu,trained=None):
        job=copy.deepcopy(refs[s]['job'])
        if trained:job.update(weights=trained['weights'],bank=trained['bank'],raw_signature=trained['signature'],base_signature=train['raw_signature'],background_method=m,side=trained['side'])
        value=launch(f'eval_{m}_{s}',job,gpu);attempt=Path(value['attempt']);metric=ab.read(attempt/'metrics.json');ref=ab.read(refs[s]['metrics'])
        old.validate_eval(tag.evidence.checked_copy(metric)[0],tag.evidence.checked_copy(ref)[0],job['val_ids'],ref['proposal_count_after'],.2)
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
        if m=='historical':
            old.require_image_replay(refs[s]['reference']['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=ref['confusion_matrix']:raise ValueError('Historical replay mismatch')
        result=dict(miou=tag.evidence.checked_copy(metric)[0]['mIoU'],metrics=str(attempt/'metrics.json'),per_image=str(attempt/'per_image_confusion.npz'))
        if trained:result.update(trained)
        with guard:results[f'{m}_{s}']=result;flush()
    def fatal(item,exc):raise exc
    try:
        flush();stages.verify_all()
        queue.run_round(list(old.SEEDS),opts.gpus,lambda s,gpu:evaluate('historical',s,gpu),cancel,deadline,fatal)
        geometry=launch('geometry',dict(train,stage='geometry'),opts.gpus[0]);side=ab.read(Path(geometry['attempt'])/'geometry.json')['tile_side_tokens']
        shared=dict(train,side=side,cache_output=str(cache),active_classes=active_classes)
        launch('probe',dict(shared,stage='probe',ids=train['train_ids'][:1]),opts.gpus[0])
        queue.run_round([0,1],opts.gpus,lambda lane,gpu:launch(f'extract_{lane}',dict(shared,stage='extract',ids=train['train_ids'][lane::2]),gpu),cancel,deadline,fatal)
        merged=stages.run('merge',lambda attempt:merge_banks(train,cache,raw_root,side,attempt))
        launch('background_diagnostic',dict(shared,stage='diagnostic'),opts.gpus[0])
        def execute(item,gpu):
            m,s=item;job=copy.deepcopy(train);sig=bank_signature(train['raw_signature'],m,side)
            job.update(stage='train',seed=s,raw=merged['raw'][m],raw_signature=sig,base_signature=train['raw_signature'],background_method=m,side=side)
            trained=launch(f'train_{m}_{s}',job,gpu);trained.update(signature=sig,side=side)
            evaluate(m,s,gpu,trained)
        def local_failure(item,exc):
            if isinstance(exc,queue.TaskOOM):failures[str(item)]='OOM; no auto retry'
            else:failures[str(item)]=str(exc)
            with guard:flush()
        queue.run_round([(m,s) for s in old.SEEDS for m in GROUPS],opts.gpus,execute,cancel,deadline,local_failure)
        gmp.verify(records,'Historical inputs');gmp.verify(identity['code'],'Frozen experiment source');stages.verify_all()
        status='complete' if len(results)==12 else 'partial'
    except queue.Deferred as exc:status='deferred';error=str(exc)
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(raw_root),str(ROOT/'logs'/opts.version),str(ROOT/'weights/sam3_adapter'/opts.version),str(ROOT/'storage/mlp_banks/voc'/opts.version)],foreground_raw_files_read_only=True,automatic_deletion=False))
        queue.__file__=oldentry
        for s,h in handlers.items():signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(opts.version,status,str(output),time.time()-started)


if __name__=='__main__':main()
