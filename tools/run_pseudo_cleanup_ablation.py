"""Clean completed semantic labels, reuse frozen tokens/proposals, retrain and evaluate."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import run_mainline_full as full


def checked_source(manifest, path):
    path = Path(path).resolve()
    record = manifest['reused_artifacts'].get(str(path))
    if record is None or full.identity(path) != record:
        raise ValueError(f'Inherited artifact changed or unregistered: {path}')
    return path


def region_reuse_plan(old_regions, new_regions):
    """A GMP vector depends on the selected token indices, not other mask pixels."""
    from src.hard_center_gmp import select_centers
    keys = {}
    for index, row in enumerate(old_regions):
        indices, _ = select_centers(row['mask'])
        keys.setdefault((int(row['cls_id']), tuple(indices.tolist())), index)
    result = []
    for row in new_regions:
        indices, _ = select_centers(row['mask'])
        result.append(keys.get((int(row['cls_id']), tuple(indices.tolist()))))
    return result


def clean_images(manifest, ids, output):
    import numpy as np
    from PIL import Image
    from src.proposal_cleanup import CleanupConfig
    from src.pseudo_label_cleanup import clean_semantic_labels
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = CleanupConfig(**manifest['label_cleanup'])
    for pos, stem in enumerate(ids, 1):
        full.free_space(manifest['root'])
        source = checked_source(manifest, Path(manifest['source_pseudo']) / f'{stem}.png')
        binding = dict(source=full.identity(source), config=config.to_dict(),
                       policy=manifest['cleanup_policy'], code=manifest['code_digest'])
        target = output / f'{stem}.png'
        if not full.checked_checkpoint(target, binding):
            with Image.open(source) as im:
                cleaned, stats = clean_semantic_labels(np.asarray(im), config)
            temporary = target.with_suffix('.tmp.png')
            Image.fromarray(cleaned).save(temporary); os.replace(temporary, target)
            full.write(target.with_suffix('.json'), dict(binding=binding,
                       artifact=full.identity(target), statistics=stats))
        print(f'[Label cleanup] {pos}/{len(ids)} {stem}', flush=True)


def build_cached_images(manifest, ids, pseudo, output, check_all_vectors=False):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from src.cached_gmp_extractor import CachedGMPExtractor
    from src.region_extractors import get_regions_from_pseudo
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = CachedGMPExtractor(Path(manifest['source_features'])/'train',
        manifest['raw_signature'], 'cuda:0', manifest['reused_artifacts'])
    totals = dict(images=0, resumed_images=0, copied_images=0,
                  reused_region_vectors=0, recomputed_region_vectors=0, max_probe_error=0.0)
    for pos, stem in enumerate(ids, 1):
        full.free_space(manifest['root'])
        image = Path(manifest['dataset'])/'JPEGImages'/f'{stem}.jpg'
        mask = Path(pseudo)/f'{stem}.png'
        old_mask = checked_source(manifest, Path(manifest['source_pseudo'])/f'{stem}.png')
        old_cache = checked_source(manifest, Path(manifest['source_region_cache'])/f'{stem}.npz')
        receipt = checked_source(manifest, old_cache.with_suffix('.json'))
        old = full.read(receipt)
        if (old['binding']['image'] != full.identity(image)
                or old['binding']['mask'] != full.identity(old_mask)
                or old['binding']['signature'] != manifest['raw_signature']
                or old['artifact'] != full.identity(old_cache)):
            raise ValueError('Source region checkpoint binding differs')
        binding = dict(image=old['binding']['image'], mask=full.identity(mask),
                       signature=manifest['raw_signature'], code=manifest['code_digest'],
                       source_checkpoint=old['artifact'])
        target = output/f'{stem}.npz'
        if full.checked_checkpoint(target, binding):
            stats = full.read(target.with_suffix('.json'))['reuse']
            totals['resumed_images'] += 1
        else:
            with np.load(old_cache, allow_pickle=False) as data:
                x = data['features']; cids = data['class_ids']
            if x.shape != (len(cids), 1280) or not np.isfinite(x).all():
                raise ValueError('Invalid inherited region vectors')
            with Image.open(mask) as a, Image.open(old_mask) as b:
                unchanged = np.array_equal(np.asarray(a), np.asarray(b))
            regions = get_regions_from_pseudo(str(image), 300, pseudo)
            if unchanged:
                if list(cids) != [r['cls_id'] for r in regions]:
                    raise ValueError('Source checkpoint class order differs')
                vectors = list(x); reused = len(vectors)
            else:
                old_regions = get_regions_from_pseudo(str(image), 300, manifest['source_pseudo'])
                if list(cids) != [r['cls_id'] for r in old_regions]:
                    raise ValueError('Source region checkpoint class order differs')
                plan = region_reuse_plan(old_regions, regions)
                missing = [i for i, source_index in enumerate(plan) if source_index is None]
                vectors = [x[i] if i is not None else None for i in plan]
                reused = len(regions)-len(missing)
                if missing:
                    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
                    model.set_image_source(image, rgb)
                    computed = model.extract_features_batch(rgb, [regions[i]['mask'] for i in missing])
                    for i, value in zip(missing, computed): vectors[i] = value
            array = np.asarray(vectors, np.float32).reshape(len(regions), 1280)
            if not np.isfinite(array).all() or (len(array) and (np.linalg.norm(array, axis=1)<1e-6).any()):
                raise ValueError('Invalid cleaned region GMP vector')
            if check_all_vectors:
                rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
                model.set_image_source(image, rgb)
                expected = np.asarray(model.extract_features_batch(rgb, [r['mask'] for r in regions]), np.float32).reshape(-1,1280)
                error = float(np.abs(expected-array).max()) if len(array) else 0.0
                totals['max_probe_error'] = max(totals['max_probe_error'], error)
                if not np.allclose(expected, array, rtol=1e-6, atol=1e-6):
                    raise ValueError('Reused and directly recomputed GMP vectors differ')
            full.save_npz(target, features=array,
                class_ids=np.asarray([r['cls_id'] for r in regions], np.int16),
                sample_ids=np.asarray([f'{stem}_{i:06d}' for i in range(len(regions))]))
            stats = dict(copied_image=unchanged, reused_regions=reused,
                         recomputed_regions=len(regions)-reused)
            full.write(target.with_suffix('.json'), dict(binding=binding,
                artifact=full.identity(target), regions=len(regions), reuse=stats))
        totals['images'] += 1; totals['copied_images'] += int(stats['copied_image'])
        totals['reused_region_vectors'] += stats['reused_regions']
        totals['recomputed_region_vectors'] += stats['recomputed_regions']
        print(f'[Cached GMP] {pos}/{len(ids)} {stem} reuse={stats["reused_regions"]} recompute={stats["recomputed_regions"]}', flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return totals


def cached_inference(manifest, argv):
    import torch
    import run_inference
    from configs import config as cfg
    from src.cached_gmp_extractor import CachedGMPExtractor
    from src.feature_adapter import AdaptedFeatureExtractor
    def builder(device, **_):
        base = CachedGMPExtractor(Path(manifest['source_features'])/'val',
            manifest['raw_signature'], device, manifest['reused_artifacts'])
        return AdaptedFeatureExtractor(base, cfg.FEATURE_ADAPTER_CKPT) if cfg.FEATURE_ADAPTER_CKPT else base
    run_inference.build_feature_extractor = builder
    print('[Features] Read-only second-run spatial tokens; original GMP and matched new MLP', flush=True)
    run_inference.main(argv)


def probe_cached(job, manifest):
    import cv2
    import numpy as np
    import torch
    from src.cached_gmp_extractor import CachedGMPExtractor
    from src.feature_bank_storage import save_feature_matrix_npz
    from src.feature_signature import save_feature_signature
    from src.build_feature_bank import save_class_mapping
    folder=Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
    ids=job['ids']; pseudo=Path(job['pseudo']); cache=Path(job['cache'])
    clean_images(manifest, ids, pseudo)
    reuse=build_cached_images(manifest, ids, pseudo, cache, check_all_vectors=True)
    model=CachedGMPExtractor(Path(manifest['source_features'])/'train',manifest['raw_signature'],
                             'cuda:0',manifest['reused_artifacts'])
    bank=Path(job['bank']);bank.mkdir(parents=True);save_class_mapping(21, 'configs/cls_voc21.txt')
    save_feature_signature(bank/'meta',manifest['raw_signature'])
    vectors=[[],[]];samples=[[],[]]
    for stem in ids:
        path=Path(manifest['dataset'])/'JPEGImages'/f'{stem}.jpg'
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        model.set_image_source(path,rgb)
        full_mask=np.ones(rgb.shape[:2],bool);half=full_mask.copy();half[:,half.shape[1]//2:]=False
        for cid,vector in enumerate(model.extract_features_batch(rgb,[full_mask,half])):
            vectors[cid].append(vector);samples[cid].append(f'{stem}_{cid:06d}')
    for cid,name in enumerate(['background','aeroplane']):
        save_feature_matrix_npz(bank/'features'/name/(name+'_features.npz'),np.stack(vectors[cid]),samples[cid])
    del model;gc.collect();torch.cuda.empty_cache()
    trainlist=folder/'train.txt';trainlist.write_text('\n'.join(ids)+'\n')
    from project_utils.experiment import train_command
    settings=dict(manifest['training'],epochs=1,steps_per_epoch=2,classes_per_batch=2,
                  instances_per_class=2,val_fraction=.25)
    weights=Path(job['weights']);mlp=Path(job['mlp'])
    full.run_script('tools/train_feature_adapter.py',train_command(bank,weights,mlp,trainlist,None,settings)[3:])
    evaluation=folder/'eval'
    evaljob=dict(operation='inference',manifest=job['manifest'],telemetry=str(folder/'eval_telemetry.json'),argv=['--strict-eval','--eval-limit','1',
        '--rescue','--rescue-raw-dir',manifest['source_raw'],'--output-dir',str(evaluation),
        '--results-json',str(evaluation/'metrics.json')])
    full.write(folder/'eval_job.json',evaljob)
    env=os.environ.copy();env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(weights/'adapter.pt'),
        OVERRIDE_INF_FEATURE_BANK_ROOT=str(mlp),OVERRIDE_INF_LOAD_PROPOSALS='true',
        OVERRIDE_INF_PROPOSALS_DIR=manifest['source_clean'])
    subprocess.run([sys.executable,'-u','-B',str(CODE/'tools/run_pseudo_cleanup_ablation.py'),
                    '--worker-spec',str(folder/'eval_job.json')],env=env,check=True)
    if full.read(evaluation/'metrics.json')['evaluated_count']!=1:
        raise ValueError('Strict cached HCR probe incomplete')
    full.write(folder/'probe.json',dict(status='complete',diagnostic_only=True,reuse=reuse,
        evaluation_telemetry=full.read(folder/'eval_telemetry.json'),
        note='Synthetic training class assignments; interface validation, not experimental accuracy'))


def work(job):
    manifest=full.read(job['manifest'])
    op=job['operation']
    if op=='clean':clean_images(manifest,job['ids'],job['output'])
    elif op=='build':full.write(job['stats'],build_cached_images(manifest,job['ids'],job['pseudo'],job['output']))
    elif op=='inference':
        if job.get('telemetry'):
            import torch
            torch.cuda.reset_peak_memory_stats()
        cached_inference(manifest,job['argv'])
        if job.get('telemetry'):
            full.write(job['telemetry'],dict(pid=os.getpid(),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    elif op=='probe':probe_cached(job,manifest)
    else:raise ValueError(f'Unknown cleanup worker operation: {op}')


def preflight(args):
    from src.proposal_cleanup import CleanupConfig
    from src.cached_gmp_extractor import CachedGMPExtractor
    from configs.experiment import TRAINING
    root=args.root;source=root/'experiments'/args.source_version
    previous=full.read(source/'manifest.json');summary=full.read(source/'summary.json')
    if (summary['status']!='complete' or summary['evaluated_count']!=1449 or summary['train_count']!=10582
            or summary.get('full_image_features') is None or '--edge-repair' not in previous['pseudo_extra_cli']
            or '--force-presence-one' not in previous['pseudo_extra_cli']):
        raise ValueError('Source must be the completed edge/presence experiment with full image caches')
    if previous['training']!=dict(TRAINING,seed=42) or previous['seed']!=42:
        raise ValueError('Source training recipe differs from current MLP2/v1 seed42')
    for name in ['src/hard_center_gmp.py','src/region_extractors.py','src/feature_extractor_sam3.py']:
        if full.sha(root/name)!=previous['code'][name]:
            raise ValueError(f'Mathematical region/GMP implementation changed: {name}')
    controls=[source/'manifest.json',source/'summary.json',source/'status.json']
    reused={}
    for stage in ['pseudo','build','proposals','cleanup','merge','train','eval']:
        marker=source/stage/'complete.json';controls.append(marker)
        for row in full.read(marker)['artifacts']:
            if row['path'] in reused and reused[row['path']]!=row:
                raise ValueError('Source phase receipts disagree')
            reused[row['path']]=row
    build=full.read(source/'build/complete.json')['result']
    signature_path=Path(summary['raw_bank'])/'meta/feature_signature.json'
    signature=full.read(signature_path);controls.append(signature_path)
    features=Path(summary['full_image_features'])
    # Constructor validates all actual aggregation/encoder choices without loading SAM3.
    CachedGMPExtractor(features/'train',signature,'cpu')
    for split,ids in [('train',previous['train_ids']),('val',previous['val_ids'])]:
        full.coverage([[p.stem for p in (features/split).glob('*.npz')]],ids)
        full.coverage([[p.stem for p in (features/split).glob('*.json')]],ids)
        for pos,stem in enumerate(ids,1):
            receipt=features/split/(stem+'.json');path=receipt.with_suffix('.npz')
            if full.identity(receipt)!=reused.get(str(receipt)) or path.stat().st_size!=reused[str(path)]['size']:
                raise ValueError(f'Changed/incomplete inherited token pair: {stem}')
            metadata=full.read(receipt)
            encoder={k:v for k,v in signature.items() if k not in ['aggregation','adapter']}
            if (metadata['binding']['encoder_signature']!=encoder or metadata['sha256']!=reused[str(path)]['sha256']
                    or metadata['shape']!=[72,72,1280] or metadata['binding']['dtype']!='float32'):
                raise ValueError(f'Inherited encoder signature differs: {stem}')
            if pos%2000==0 or pos==len(ids):print(f'[Preflight] {split} token receipts {pos}/{len(ids)}',flush=True)
    inputs=previous['inputs']+[full.identity(p) for p in controls]
    full.verify(inputs)
    pseudo=root/'storage/pseudo_labels/voc'/args.source_version
    full.validate_pseudo(pseudo,previous['train_ids'],Path(previous['dataset']))
    for stem in previous['train_ids']:
        for p in [pseudo/(stem+'.png'),Path(build['cache'])/(stem+'.npz'),Path(build['cache'])/(stem+'.json')]:
            if full.identity(p)!=reused.get(str(p)):
                raise ValueError(f'Inherited training asset changed: {p}')
    full.validate_proposals(summary['raw_proposals'],previous['val_ids'],previous['dataset'])
    full.validate_proposals(summary['clean_proposals'],previous['val_ids'],previous['dataset'],clean=True)
    for directory in [summary['raw_proposals'],summary['clean_proposals']]:
        for p in Path(directory).glob('*.npz'):
            if full.identity(p)!=reused.get(str(p)):
                raise ValueError('Inherited validation proposal changed')
    report=Path(summary['weights'])/'training_report.json';controls.append(report)
    if full.identity(report)!=reused.get(str(report)):
        raise ValueError('Frozen internal split report changed')
    full.free_space(root,full.RESERVE+10*2**30);gpu=full.idle_gpus(args.gpus)
    print(f'[Preflight] source complete; GPUs={gpu}; token SHA checked when consumed, no SAM3 forwards',flush=True)
    code={str(p.relative_to(root)):full.sha(p) for p in full.source_files(root)}
    return dict(format_version=1,version=args.version,root=str(root),source_version=args.source_version,
        dataset=previous['dataset'],checkpoint=previous['checkpoint'],gpus=args.gpus,seed=42,
        train_ids=previous['train_ids'],val_ids=previous['val_ids'],inputs=inputs+[full.identity(report)],
        code=code,code_digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
        training=previous['training'],label_cleanup=CleanupConfig().to_dict(),
        cleanup_policy='semantic_per_class_geometry_only_no_scores_no_NMS_background_only_hole_fill',
        source_pseudo=str(pseudo),source_features=str(features),source_region_cache=build['cache'],
        source_raw=summary['raw_proposals'],source_clean=summary['clean_proposals'],
        source_training_report=str(report),source_mIoU=summary['mIoU'],
        source_pseudo_mIoU=summary['pseudo_quality']['mIoU'],raw_signature=signature,
        reused_artifacts=reused,historical_inputs=True,pseudo_extra_cli=[])


def pipeline(launch,probe_only=False):
    from configs import config as cfg
    from tools.pseudo_mask_evaluation import evaluate_pseudo_masks
    m=launch.manifest;root=launch.args.root;folder=launch.folder;version=m['version']
    pseudo=root/'storage/pseudo_labels/voc'/version
    rawbase=root/'storage/raw_banks/voc'/version;cache=rawbase/'image_cache'
    weightsbase=root/'weights/sam3_adapter'/version;mlpbase=root/'storage/mlp_banks/voc'/version
    full.write(folder/'artifacts.json',dict(owned_roots=list(map(str,[pseudo,rawbase,weightsbase,mlpbase,
        folder,launch.logroot,root/'storage/runtime'/version])),historical_inputs=True,
        referenced_roots=[m['source_pseudo'],m['source_features'],m['source_region_cache'],m['source_raw'],m['source_clean']]))
    def jobs(stage,attempt,specs):
        wrapped=[]
        for i,spec in enumerate(specs):
            gpu=spec.pop('gpu',None);overrides=spec.pop('overrides',{})
            file=attempt/f'spec_{i}.json';full.write(file,dict(spec,manifest=str(folder/'manifest.json')))
            wrapped.append(dict(operation='script',script='tools/run_pseudo_cleanup_ablation.py',
                argv=['--worker-spec',str(file)],gpu=gpu,overrides=overrides))
        launch.jobs(stage,attempt,wrapped)
    def do_probe(attempt):
        labels=full.read(Path(m['dataset'])/'ImageSets/ImageLevel/trainaug_imagelevel.json')['images']
        byid={Path(x['file_name']).stem:x['labels'] for x in labels}
        table=next(i for i in m['train_ids'] if 11 in byid[i])
        ids=[table]+[i for i in m['train_ids'] if i!=table][:3]
        specs=[]
        for g in m['gpus']:
            suffix=Path('probe')/attempt.name/f'gpu{g}'
            specs.append(dict(operation='probe',gpu=g,ids=ids,folder=str(attempt/f'gpu{g}'),
                pseudo=str(pseudo/suffix),cache=str(rawbase/suffix/'image_cache'),
                bank=str(rawbase/suffix/'bank'),weights=str(weightsbase/suffix),mlp=str(mlpbase/suffix),
                overrides={'OVERRIDE_OUTPUT_ROOT':str(rawbase/suffix/'bank')}))
        jobs('probe',attempt,specs)
        return dict(diagnostic_only=True,status='complete',
                    probes=[full.read(attempt/f'gpu{g}/probe.json') for g in m['gpus']]),[attempt]+[
                        p/'probe'/attempt.name for p in [pseudo,rawbase,weightsbase,mlpbase]]
    launch.stage('probe',do_probe)
    if probe_only:return dict(status='ready',version=version,diagnostic_only=True,workers=launch.reports)
    def do_clean(attempt):
        jobs('pseudo_cleanup',attempt,[dict(operation='clean',ids=m['train_ids'][i::2],output=str(pseudo)) for i in range(2)])
        full.validate_pseudo(pseudo,m['train_ids'],m['dataset'])
        totals={k:0 for k in ['changed_images','changed_pixels','removed_island_pixels','proposed_hole_pixels',
            'filled_background_pixels','protected_other_class_pixels','area_rejected_classes']}
        for stem in m['train_ids']:
            stats=full.read(pseudo/(stem+'.json'))['statistics']
            totals['changed_images']+=int(stats['changed'])
            for k in totals:
                if k!='changed_images':totals[k]+=stats[k]
        full.write(attempt/'cleanup_statistics.json',totals)
        return dict(path=str(pseudo),count=len(m['train_ids']),statistics=totals),[pseudo,attempt]
    cleaned=launch.stage('pseudo_cleanup',do_clean)
    def do_quality(attempt):
        samples=[dict(image=Path(m['dataset'])/'JPEGImages'/(i+'.jpg'),prediction=pseudo/(i+'.png')) for i in m['train_ids']]
        report=attempt/'miou_trainaug.json'
        result=evaluate_pseudo_masks(samples,m['dataset'],'voc','trainaug',cfg.DATASET_CLASSES,
                                    report,gt_dir=Path(m['dataset'])/'SegmentationClassAug')
        if result['status']!='complete' or result['evaluated_count']!=10582:raise ValueError('Incomplete cleaned label evaluation')
        return dict(metrics=str(report),mIoU=result['mIoU']),[attempt]
    quality=launch.stage('pseudo_eval',do_quality)
    def do_build(attempt):
        jobs('build',attempt,[dict(operation='build',gpu=g,ids=m['train_ids'][i::2],pseudo=str(pseudo),
             output=str(cache),stats=str(attempt/f'reuse_{i}.json')) for i,g in enumerate(m['gpus'])])
        full.coverage([[p.stem for p in cache.glob('*.npz')]],m['train_ids'])
        signature=attempt/'signature.json';full.write(signature,m['raw_signature'])
        return dict(cache=str(cache),signatures=[str(signature)],reuse=[full.read(attempt/f'reuse_{i}.json') for i in range(2)]),[cache,attempt]
    extracted=launch.stage('build',do_build)
    def do_merge(attempt):
        bank=rawbase/'bank'/attempt.name
        launch.jobs('merge',attempt,[dict(operation='merge',cache=str(cache),ids=m['train_ids'],signatures=extracted['signatures'],
            bank=str(bank),overrides={'OVERRIDE_OUTPUT_ROOT':str(bank)})])
        return dict(path=str(bank)),[bank,attempt]
    bank=Path(launch.stage('merge',do_merge)['path'])
    def do_train(attempt):
        from project_utils.experiment import train_command
        weights=weightsbase/'42'/attempt.name;mlp=mlpbase/'42'/attempt.name
        launch.jobs('train',attempt,[dict(operation='script',script='tools/train_feature_adapter.py',gpu=m['gpus'][1],
            argv=train_command(bank,weights,mlp,Path(m['dataset'])/'ImageSets/Segmentation/trainaug.txt',
                m['source_training_report'],m['training'])[3:])])
        report=full.read(weights/'training_report.json');previous=full.read(m['source_training_report'])
        if (len(report['epochs'])!=50 or report['args']['num_layers']!=2
                or report['validation_images']!=previous['validation_images']
                or report['train_images']!=previous['train_images']):
            raise ValueError('MLP training incomplete or internal image split changed')
        return dict(weights=str(weights),bank=str(mlp),best_epoch=report['best_epoch']),[weights,mlp,attempt]
    trained=launch.stage('train',do_train)
    def do_eval(attempt):
        jobs('eval',attempt,[dict(operation='inference',gpu=g,
            argv=['--strict-eval','--eval-limit','0','--shard-count','2','--shard-index',str(i),
                  '--rescue','--rescue-raw-dir',m['source_raw'],'--output-dir',str(attempt/f'gpu{g}'),
                  '--results-json',str(attempt/f'gpu{g}/metrics.json')],
            overrides={'OVERRIDE_FEATURE_ADAPTER_CKPT':str(Path(trained['weights'])/'adapter.pt'),
                'OVERRIDE_INF_FEATURE_BANK_ROOT':trained['bank'],'OVERRIDE_INF_LOAD_PROPOSALS':'true',
                'OVERRIDE_INF_PROPOSALS_DIR':m['source_clean']}) for i,g in enumerate(m['gpus'])])
        metrics=full.merge_metrics([full.read(attempt/f'gpu{g}/metrics.json') for g in m['gpus']],m['val_ids'])
        full.write(attempt/'metrics.json',metrics)
        with (attempt/'per_class.csv').open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['class','IoU','IoU_percent'])
            for name,value in metrics['per_class_iou'].items():writer.writerow([name,value,value*100])
        return dict(metrics=str(attempt/'metrics.json'),per_class=str(attempt/'per_class.csv')),[attempt]
    evaluated=launch.stage('eval',do_eval)
    for key,name in [('metrics','metrics.json'),('per_class','per_class.csv')]:
        target=folder/name
        if target.exists():
            if full.sha(target)!=full.sha(evaluated[key]):raise ValueError('Published result changed')
        else:shutil.copy2(evaluated[key],target)
    metrics=full.read(folder/'metrics.json')
    return dict(status='complete',version=version,seed=42,mIoU=metrics['mIoU'],per_class_iou=metrics['per_class_iou'],
        evaluated_count=metrics['evaluated_count'],train_count=10582,pseudo_quality=quality,best_epoch=trained['best_epoch'],
        region_statistics=full.read(bank/'meta/filter_report.json'),workers=launch.reports,cleanup=cleaned,
        raw_bank=str(bank),mlp_bank=trained['bank'],weights=trained['weights'],historical_inputs=True,
        source_version=m['source_version'],source_mIoU=m['source_mIoU'],delta_mIoU=metrics['mIoU']-m['source_mIoU'],
        source_pseudo_mIoU=m['source_pseudo_mIoU'],delta_pseudo_mIoU=quality['mIoU']-m['source_pseudo_mIoU'],
        reused_full_image_features=m['source_features'],raw_proposals=m['source_raw'],clean_proposals=m['source_clean'],
        feature_reuse=extracted['reuse'],cleanup_policy=m['cleanup_policy'],label_cleanup=m['label_cleanup'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=CODE);parser.add_argument('--version')
    parser.add_argument('--source-version',default='mainline_edge_presence1_features_20261010_224233')
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--worker-spec',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_spec:work(full.read(args.worker_spec));return
    if not args.version or not all(c.isalnum() or c in '_-' for c in args.version) or len(set(args.gpus))!=2:
        parser.error('Provide a safe new version and two distinct physical GPUs')
    args.root=args.root.resolve()
    for key in list(os.environ):
        if key.startswith('OVERRIDE_'):del os.environ[key]
    folder=args.root/'experiments'/args.version
    if folder.exists() and not args.resume:raise FileExistsError('Choose a new experiment version')
    if args.resume and not (folder/'manifest.json').is_file():raise ValueError('Missing resume manifest')
    m=preflight(args)
    if args.preflight_only:print('[Preflight] passed; not started',flush=True);return
    if args.resume:
        if full.read(folder/'manifest.json')!=m:raise ValueError('Resume code, parameters or input signatures changed')
    else:
        for parent in ['storage/pseudo_labels/voc','storage/raw_banks/voc','storage/mlp_banks/voc','weights/sam3_adapter','logs']:
            if (args.root/parent/args.version).exists():raise FileExistsError('Historical output path already exists')
        folder.mkdir(parents=True);full.write(folder/'manifest.json',m)
    lock=folder/'launcher.lock'
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream:stream.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    started=time.monotonic();launch=None;summary=None
    def interrupted(signum,_):raise KeyboardInterrupt(f'Received signal {signum}')
    old_signal=signal.signal(signal.SIGTERM,interrupted)
    try:
        code=folder/'code'
        if not code.exists():
            code.mkdir()
            for relative in m['code']:
                target=code/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.root/relative,target)
        if any(full.sha(code/p)!=digest for p,digest in m['code'].items()):raise ValueError('Immutable code snapshot differs')
        sys.path[:0]=[str(code),str(code/'sam3-main')]
        runtime=args.root/'storage/runtime'/args.version
        for name in ['tmp','huggingface','torch','xdg']:(runtime/name).mkdir(parents=True,exist_ok=True)
        os.environ.update(TMPDIR=str(runtime/'tmp'),TEMP=str(runtime/'tmp'),TMP=str(runtime/'tmp'),
            HF_HOME=str(runtime/'huggingface'),TORCH_HOME=str(runtime/'torch'),XDG_CACHE_HOME=str(runtime/'xdg'))
        launch=full.Launcher(args,m,folder,code);summary=pipeline(launch,args.probe_only)
    except BaseException as error:
        summary=dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',version=args.version,
            failed_stage=launch.current if launch else 'initialization',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        summary=summary or dict(status='failed',version=args.version,failed_stage='initialization')
        summary['seconds']=time.monotonic()-started;full.write(folder/'summary.json',summary)
        full.write(folder/'status.json',dict(status=summary['status'],stage=summary.get('failed_stage','complete'),
                                           seconds=summary['seconds'],finished=time.time()))
        if summary['status']!='ready':
            from project_utils.experiment_notify import notify_experiment
            files=[p for p in [folder/'summary.json',folder/'metrics.json',folder/'per_class.csv'] if p.is_file()]
            accepted=False
            for attempt in range(3):
                accepted=notify_experiment(args.version,summary['status'],str(folder),summary['seconds'],summary=summary,attachments=files)
                if accepted:break
                if attempt<2:time.sleep(10*(attempt+1))
            full.write(folder/'notification.json',dict(smtp_accepted=accepted,attempts=attempt+1,
                attachments=[p.name for p in files],inbox_delivery_confirmed=False))
        lock.unlink();signal.signal(signal.SIGTERM,old_signal)
    if summary['status'] not in ['ready','complete']:raise SystemExit(1)


if __name__=='__main__':main()
