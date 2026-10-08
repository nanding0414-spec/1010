"""Frozen real-bank GMP/MLP experiment with synthetic camera-view training only.

No generated directory is needed for --help/tests. Use --inventory-only while
generation is running; a formal run requires all L30/R30/L60/R60 trainaug views.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_bank_provenance as provenance
from tools.experimental_background_partition_recovery import AliasAwareStages
from tools.experimental_tag_constraint_recovery import evaluation_environment

ab, queue, gmp, old = provenance.ab, provenance.queue, provenance.gmp, provenance.old
PROTOCOL = 'synthetic_view_training_real_gallery_v1'
METHODS = ('baseline', 'view30', 'view_all', 'view_weighted')
VIEWS = ('L30', 'R30', 'L60', 'R60')
OLD_ROOT = '/home/zystu2025/nd/ModuSeg_SAM3'
ACTIVE = dict(preview=True, preview_resolution=672, presence_threshold=.2,
              active_keep_ratio=.4, min_active_classes=1, max_active_classes=0,
              bucket_size=4, neighbor_tokens=2, hypernym_tokens=2, context_strength=1.,
              confidence_threshold=.5, mask_nms_threshold=.7, min_mask_area=100,
              min_mask_region_area=0, max_mask_area_ratio=0., max_masks=0)
RULE = dict(generated_foreground_only=True, generated_per_four_foreground=2,
            angle_sampling_weights={'view30': [1, 1, 0, 0], 'view_all': [1, 1, 1, 1],
                                    'view_weighted': [1, 1, .25, .25]},
            min_region_area=300, rho=.1, seeds=[42, 123, 2026],
            epochs=50, steps=100, hcr=False, export='original real members only',
            internal_validation='original real images only')


def relocate(value):
    """Rebase exact old-project prefixes in memory, never edit historical bytes."""
    if isinstance(value, str) and (value == OLD_ROOT or value.startswith(OLD_ROOT+'/')):
        return str(ROOT / value[len(OLD_ROOT):].lstrip('/'))
    if isinstance(value, list): return [relocate(v) for v in value]
    if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
    return value


def install_relocation():
    original = ab.read
    def read(path): return relocate(original(relocate(str(path))))
    ab.read = read


def unique_records(records):
    result = {}; resolved = {}
    for record in records:
        path=record['path']
        if path not in resolved: resolved[path]=str(Path(relocate(path)).resolve())
        item = dict(record, path=resolved[path])
        if item['path'] in result and result[item['path']] != item:
            raise ValueError('Conflicting SHA identities: '+item['path'])
        result[item['path']] = item
    return [result[k] for k in sorted(result)]


def inventory(directory, train_ids, val_ids, hash_files=False):
    """Strict filename/source membership; inventory tolerates missing views only."""
    from PIL import Image
    directory = Path(directory).resolve()
    if not directory.is_dir(): raise FileNotFoundError(directory)
    train, val = set(train_ids), set(val_ids)
    if len(train) != len(train_ids) or len(val) != len(val_ids) or train & val:
        raise ValueError('Duplicate/overlapping train and VOC val IDs')
    files, ignored, foreign = {}, [], []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            continue
        match = re.fullmatch(r'(.+)_(L30|R30|L60|R60)', path.stem, re.IGNORECASE)
        stem, view = (match[1], match[2].upper()) if match else (path.stem, 'original')
        if stem in val: raise ValueError('VOC val image/descendant in generation folder: '+str(path))
        if stem not in train:
            foreign.append(str(path)); continue
        if view == 'original':
            ignored.append(str(path)); continue  # authoritative original remains the historical VOC image
        key = stem+'_'+view
        if key in files: raise ValueError('Duplicate generated view: '+key)
        row = dict(source_id=stem, view=view, path=str(path.resolve()))
        if hash_files:
            with Image.open(path) as image:
                image.load()
                if min(image.size) < 14: raise ValueError('Generated image too small: '+str(path))
                row['size_wh'] = list(image.size)
            row.update(sha256=ab.digest(path), bytes=path.stat().st_size)
        files[key] = row
    if foreign: raise ValueError('Unknown image names (not trainaug or supported angle): '+repr(foreign[:10]))
    missing = [stem+'_'+v for stem in sorted(train) for v in VIEWS if stem+'_'+v not in files]
    return dict(root=str(directory), files=files, missing=missing, complete=not missing,
                counts=dict(Counter(r['view'] for r in files.values())),
                ignored_original_copies=len(ignored), expected=len(train)*4)


def verify_aug_list(path, train_ids, inv):
    lines=Path(path).read_text(encoding='utf-8-sig').splitlines()
    names=[]
    for line in lines:
        parts=line.strip().split()
        if len(parts)!=1: raise ValueError('Augmented list must have one image ID per line')
        names.append(Path(parts[0]).stem)
    expected=set(train_ids)|set(inv['files'])
    if len(names)!=len(set(names)) or set(names)!=expected:
        raise ValueError(f'Augmented list differs: missing={sorted(expected-set(names))[:5]}, extra={sorted(set(names)-expected)[:5]}, duplicates={len(names)-len(set(names))}')
    return dict(path=str(Path(path).resolve()),count=len(names),sha256=ab.digest(path))


def pseudo_map(masks, classes, allowed, shape):
    masks, classes = np.asarray(masks), np.asarray(classes)
    if masks.shape != (len(classes), *shape) or not np.isin(masks, [0, 1]).all():
        raise ValueError('Invalid generated masks')
    if not np.isin(classes, allowed).all(): raise ValueError('Active returned disallowed class')
    result = np.zeros(shape, np.uint8)
    for c in allowed:
        union = masks[classes == c].astype(bool).any(axis=0)
        result[union & (result == 0)] = c
    return result


def choose_aug_rows(rows, groups, views, excluded, count, rng, method):
    """Choose an angle, then a source image, then a region; avoid family duplicates."""
    weights = np.asarray(RULE['angle_sampling_weights'][method], np.float64)
    result = []; excluded = set(excluded)
    for _ in range(count):
        available = rows[~np.isin(groups[rows], list(excluded))]
        pools = [available[views[available] == v] for v in VIEWS]
        probability = weights * np.asarray([bool(len(p)) for p in pools])
        if probability.sum() == 0: break
        angle = int(rng.choice(4, p=probability/probability.sum()))
        pool = pools[angle]; family = rng.choice(np.unique(groups[pool]))
        selected = int(rng.choice(pool[groups[pool] == family]))
        result.append(selected); excluded.add(str(family))
    return result


def augmented_batch(original, y, groups, train, rng, classes_per_batch,
                    instances_per_class, pools, views, bg_index, method, counts):
    rows = original(y, groups, train, rng, classes_per_batch, instances_per_class)
    if method == 'baseline': return rows
    result = []
    for cls in dict.fromkeys(y[rows].tolist()):
        real = rows[y[rows] == cls]
        if cls == bg_index:
            result.extend(real); counts['real_background'] += len(real); continue
        keep = real[:max(1, len(real)//2)]
        selected = choose_aug_rows(pools[cls], groups, views, groups[keep],
                                   len(real)-len(keep), rng, method)
        # A rare source class can lack enough distinct generated families. Retain
        # original rows explicitly, never silently duplicate a synthetic vector.
        rest = [r for r in real[len(keep):] if groups[r] not in set(groups[selected])]
        fill = rest[:len(real)-len(keep)-len(selected)]
        result.extend([*keep, *selected, *fill])
        counts['real_foreground'] += len(keep)+len(fill)
        counts['synthetic_shortfall'] += len(real)-len(keep)-len(selected)
        for r in selected: counts[str(views[r])] += 1
    return np.asarray(result, np.int64)


def history(previous):
    print('[History] validating moved paths against unchanged SHA256 records', flush=True)
    manifest, summary = ab.read(previous/'manifest.json'), ab.read(previous/'summary.json')
    if summary['status'] != 'complete' or not summary['groups']['historical']['complete']:
        raise ValueError('Need completed background-partition clean reference')
    gmp.verify(unique_records(manifest['code']), 'Historical source')
    records = unique_records(manifest['records']); del manifest
    gmp.verify(records, 'Historical inputs')
    prepared_path = provenance.DEPLOY/'prepared.json'
    prepared = ab.read(prepared_path); refs = {}; split = None
    keys = ('dataset_root image_dir checkpoint train_list val_list imagelevel_json gt_dir images train_ids val_ids '
            'source cache cache_identity checkpoint_sha256 pseudo proposals raw_signature weights bank cleanup '
            'stage group seed role variant').split()
    for seed in old.SEEDS:
        source = prepared['jobs'][f'baseline_{seed}']
        job = {k: copy.deepcopy(source['job'][k]) for k in keys}
        ref = summary['results'][f'historical_{seed}']
        marker = previous/f'eval_historical_{seed}'/'complete.json'
        artifacts = unique_records(ab.read(marker)['artifacts'])
        gmp.verify(artifacts, 'Historical replay'); records += artifacts+ab.identities([marker])
        metric, replay = ab.read(source['metrics']), ab.read(ref['metrics'])
        if metric['confusion_matrix'] != replay['confusion_matrix']: raise ValueError('Baseline matrices differ')
        old.require_image_replay(source['reference']['per_image'], ref['per_image'])
        if (job['role'], job['group'], job['variant']) != ('gmp2', 'C', 'd2'):
            raise ValueError('Need hard-center GMP rho=.1 and two-layer MLP')
        report_path = Path(job['weights'])/'training_report.json'; report = ab.read(report_path)
        current_split = {k: report[k] for k in ('train_images', 'validation_images')}
        if split is not None and split != current_split: raise ValueError('Historical seed splits differ')
        split = current_split
        job.update(raw=report['args']['source_bank'])
        if ab.read(Path(job['raw'])/'meta/feature_signature.json') != job['raw_signature']:
            raise ValueError('Raw GMP source signature differs')
        refs[seed] = dict(job=job, environment=source['command']['environment'],
                          metrics=ref['metrics'], per_image=ref['per_image'])
    del prepared
    if not split['train_images'] or not split['validation_images'] or set(split['train_images']) & set(split['validation_images']):
        raise ValueError('Invalid historical internal split')
    recorded = {r['path'] for r in records}
    used = ab.files_under([Path(refs[s]['job'][k]) for s in old.SEEDS for k in ('raw', 'weights', 'bank')])
    used += [Path(refs[42]['job'][k]) for k in ('checkpoint', 'imagelevel_json', 'train_list', 'val_list')]
    if any(str(p.resolve()) not in recorded for p in used): raise ValueError('Historical file has no trusted SHA record')
    first = refs[42]['job']
    if len(first['val_ids']) != 1449 or len(set(first['val_ids'])) != 1449: raise ValueError('Expected full VOC val')
    if set(first['train_ids']) != set(Path(first['train_list']).read_text().split()): raise ValueError('Train IDs differ')
    if not (set(split['train_images']) | set(split['validation_images'])) <= set(first['train_ids']):
        raise ValueError('Internal split outside trainaug')
    for ref in refs.values():
        if any(ref['job'][k] != first[k] for k in ('raw', 'cache', 'cache_identity', 'val_ids', 'proposals')):
            raise ValueError('Seeds do not share inputs')
    for stem in first['val_ids']:
        marker = Path(first['cache'])/'val'/stem/'complete.json'
        cached = queue.cache_read(marker.parent, first['cache_identity'][stem])
        if cached is None: raise ValueError('Missing query cache: '+stem)
        records += cached['artifacts']+ab.identities([marker])
    records += ab.identities([previous/'manifest.json', previous/'summary.json', prepared_path])
    return refs, split, unique_records(records)


def cache_identity(job, key):
    return dict(protocol=PROTOCOL, input=job['view_files'][key], allowed=job['tags'][job['view_files'][key]['source_id']],
                active=ACTIVE, vocabulary=job['active_classes'], rule=RULE,
                signature=job['raw_signature'], source_sha=job['source_sha'], precision=ab.POLICY)


def read_vectors(job, key):
    cached = queue.cache_read(Path(job['view_cache'])/key, cache_identity(job, key))
    if cached is None: raise ValueError('Missing generated region cache: '+key)
    with np.load(cached['file'], allow_pickle=False) as data:
        vectors, labels = data['vectors'].copy(), data['labels'].copy()
    if vectors.shape != (len(labels), 1280) or not np.isfinite(vectors).all() or not np.isin(labels, range(1,21)).all():
        raise ValueError('Invalid generated region features')
    if len(vectors) and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
        raise ValueError('Invalid generated vector normalization')
    return vectors, labels, cached


def extract_worker(job):
    import torch
    from PIL import Image
    from src.feature_extractor_factory import build_feature_extractor
    from src.region_extractors import _extract_class_regions
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    from tools.experimental_background_partition_queue import pool
    model = build_feature_extractor(torch.device('cuda:0'), use_interactivity=False)
    if model.feature_signature() != job['source']: raise ValueError('Live feature signature differs')
    vocabulary = SAM3PseudoMaskGenerator(dataset_type='voc', load_model=False, output_dir=job['view_cache'])
    if old.canonical([asdict(c) for c in vocabulary.active_classes]) != job['active_classes']:
        raise ValueError('Active vocabulary changed')
    proposer = SAM3ActiveProposer(model.model, torch.device('cuda:0'), vocabulary.active_classes, **ACTIVE)
    # Before new data: reproduce one original real-region vector using live tokens.
    if job['stage'] == 'probe':
        from src.region_extractors import get_regions_from_pseudo
        stem = job['probe_id']; rows = ab.read(job['plan'])[stem]
        regions = get_regions_from_pseudo(job['images'][stem], 300, job['pseudo'])
        masks = [regions[r['index']]['mask'] for r in rows]
        if queue.mask_hashes(masks) != [r['mask_sha256'] for r in rows]: raise ValueError('Original masks changed')
        previous = queue.cache_vectors(Path(job['cache'])/'train'/stem, job['cache_identity'][stem], 'C', queue.mask_hashes(masks))
        with Image.open(job['images'][stem]) as image: rgb = np.asarray(image.convert('RGB'))
        with torch.inference_mode():
            tokens, grid = model._forward_tokens(model._preprocess_image(rgb)[0])
            reproduced = np.stack([pool(model, tokens, mask)[0] for mask in masks])
        if not np.allclose(previous, reproduced, atol=5e-5, rtol=5e-5): raise ValueError('Original GMP reproduction failed')
        ab.write(Path(job['attempt'])/'probe.json', dict(original_max_abs=float(np.max(np.abs(previous-reproduced)))))
        del tokens
    for n, key in enumerate(job['ids'], 1):
        queue.ensure_resources(job['deadline'])
        row = job['view_files'][key]
        if ab.digest(row['path']) != row['sha256']: raise ValueError('Generated image changed: '+key)
        folder = Path(job['view_cache'])/key
        if (folder/'complete.json').exists(): read_vectors(job, key); continue
        started = time.monotonic()
        with Image.open(row['path']) as image: rgb = np.asarray(image.convert('RGB'))
        allowed = job['tags'][row['source_id']]
        with torch.inference_mode():
            output = proposer(np.ascontiguousarray(rgb[...,::-1]), class_indices=[v-1 for v in allowed])
            masks = output['instances'].pred_masks.cpu().numpy()
            labels = np.asarray(proposer.last_result['class_ids'])
            seg = pseudo_map(masks, labels, allowed, rgb.shape[:2])
            regions = [r for r in _extract_class_regions(seg, 300) if r['cls_id'] != 0]
            values = []; diagnostics = []
            if regions:
                tokens, grid = model._forward_tokens(model._preprocess_image(rgb)[0])
                if tuple(grid) != (72,72) or tokens.shape != (1,5184,1280): raise ValueError('Token geometry changed')
                for region in regions:
                    vector, info = pool(model, tokens, region['mask']); values.append(vector); diagnostics.append(info)
                del tokens
        attempt = queue.next_attempt(folder); file = attempt/'regions.npz'
        vectors = np.asarray(values, np.float32).reshape(-1,1280)
        np.savez_compressed(file, vectors=vectors, labels=np.asarray([r['cls_id'] for r in regions], np.int64))
        # PNG is an audit artifact, not an alteration of original pseudo labels.
        pseudo = Path(job['view_pseudo'])/key/attempt.name/'pseudo.png'
        pseudo.parent.mkdir(parents=True,exist_ok=False); Image.fromarray(seg).save(pseudo)
        ab.write(attempt/'report.json', dict(source_id=row['source_id'], view=row['view'],
            allowed=allowed, detected=sorted(set(labels.tolist())), region_count=len(regions),
            mask_hashes=queue.mask_hashes([r['mask'] for r in regions]), diagnostics=diagnostics,
            seconds=time.monotonic()-started, synthetic_background_used=False))
        ab.write(folder/'complete.json', dict(identity=cache_identity(job,key), file=str(file),
            artifacts=ab.identities([file,pseudo,attempt/'report.json'])))
        if not np.array_equal(read_vectors(job,key)[0], vectors): raise ValueError('Cache roundtrip differs')
        print(f'[Views] {n}/{len(job["ids"])} {key}: {len(regions)} foreground regions', flush=True)
        del output, masks; proposer.last_result = None


def install_training(job):
    """Keep the exact v1 trainer/loss/export; change only training minibatch input.

    The trainer sees original-only train/val index arrays. The augmented sampler
    may select appended rows, but validation galleries and exported records remain
    real-only. This avoids changing model selection via an enlarged gallery.
    """
    import src.adapter_training as at
    originals = (at.load_training_bank, at.split_from_report, at.balanced_batch)
    state = {}; counts = Counter()
    def load(root, train_list):
        x,y,groups,classes,signature,records = originals[0](root,train_list)
        if signature != job['raw_signature']: raise ValueError('Training source signature differs')
        state.update(n=len(x), y=y.copy(), groups=groups.copy())
        features, targets, families, angles = [], [], [], []
        allowed = set(job['split']['train_images']); forbidden = set(job['split']['validation_images'])
        for key in sorted(job['view_files']):
            row = job['view_files'][key]
            if row['source_id'] not in allowed or row['source_id'] in forbidden: raise ValueError('Generated family split leakage')
            values, labels, _ = read_vectors(job,key)
            features.append(values)
            targets.extend(classes.index(provenance.evidence.VOC_CLASSES[int(c)]) for c in labels)
            families.extend([row['source_id']]*len(values)); angles.extend([row['view']]*len(values))
        if not features: raise ValueError('No generated features')
        x = np.concatenate([x,*features]); y = np.concatenate([y,np.asarray(targets,np.int64)])
        groups = np.concatenate([groups,np.asarray(families)])
        views = np.asarray(['real']*state['n']+angles)
        pools = {int(c): np.flatnonzero((y==c)&(np.arange(len(y))>=state['n'])) for c in range(len(classes))}
        bg = classes.index('background')
        for c,pool in pools.items():
            if c == bg: continue
            eligible = pool[np.isin(views[pool], VIEWS[:2])] if job['method']=='view30' else pool
            if len(np.unique(groups[eligible])) < 2 and not job.get('smoke'):
                raise ValueError('Insufficient generated training families for '+classes[c])
        state.update(pools=pools, views=views, bg=bg, classes=classes)
        return x,y,groups,classes,signature,records
    def split(groups,labels,path):
        train,val = originals[1](groups[:state['n']],labels[:state['n']],path)
        if not set(groups[train]) <= set(job['split']['train_images']): raise ValueError('Training split drift')
        return train,val
    def batch(y,groups,train,rng,classes_per_batch,instances_per_class):
        return augmented_batch(originals[2],y,groups,train,rng,classes_per_batch,instances_per_class,
                               state['pools'],state['views'],state['bg'],job['method'],counts)
    at.load_training_bank,at.split_from_report,at.balanced_batch = load,split,batch
    return counts, state


def worker(job):
    os.chdir(ROOT); sys.path.insert(0,str(ROOT/'sam3-main'))
    import torch
    started=time.monotonic(); status='failed'; category='local'; error=None
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    try:
        if job['protocol'] != PROTOCOL or job['rule'] != old.canonical(RULE): raise ValueError('Protocol mismatch')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        if job['stage'] in ('probe','extract'):
            ab.install_precision()
            from project_utils.seed_utils import set_global_seed
            set_global_seed(42); extract_worker(job)
        elif job['stage']=='train':
            counts,state=install_training(job)
            if job.get('smoke'):
                run=ab.runpy.run_path
                def short_train(path,*args,**kwargs):
                    if Path(path).resolve()!=ROOT/'tools/train_feature_adapter.py': raise ValueError('Unexpected smoke trainer')
                    for flag,value in (('--epochs','1'),('--steps-per-epoch','100')):
                        if sys.argv.count(flag)!=1: raise ValueError('Missing smoke training flag')
                        sys.argv[sys.argv.index(flag)+1]=value
                    return run(path,*args,**kwargs)
                ab.runpy.run_path=short_train
            old.previous.worker(job)
            if job.get('smoke') and sum(counts[v] for v in VIEWS)==0:
                raise ValueError('Smoke trainer did not sample any generated regions')
            if ab.bank_members(job['bank']) != ab.bank_members(job['raw']): raise ValueError('Generated members entered retrieval bank')
            report=ab.read(Path(job['weights'])/'training_report.json')
            if any(report[k]!=job['split'][k] for k in job['split']): raise ValueError('Split changed during training')
            ab.write(Path(job['attempt'])/'augmentation.json',dict(method=job['method'],counts=dict(counts),
                appended_rows=len(state['views'])-state['n'],rule=RULE,bank_members='original only'))
        elif job['stage']=='eval':
            queue.KEY_IMAGES=[]; old.previous.worker(job)
        else: raise ValueError('Unsupported stage')
        torch.cuda.synchronize(); status='complete'
    except queue.Deferred as exc: category='deferred'; error=str(exc); raise
    except torch.cuda.OutOfMemoryError as exc: category='oom'; error=str(exc); raise
    except (ValueError,KeyError,FileNotFoundError,queue.GlobalFailure) as exc: category='global'; error=str(exc); raise
    except BaseException as exc: error=str(exc); raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,category=category,error=error,
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),precision_policy=ab.POLICY))


def summarize(results):
    groups={}; contrasts={}
    for method in METHODS:
        scores={str(s):results[f'{method}_{s}']['miou'] for s in old.SEEDS if f'{method}_{s}' in results}
        complete=len(scores)==3
        groups[method]=dict(complete=complete,seeds=scores,
            mean=float(np.mean(list(scores.values()))) if complete else None,
            sample_std=float(np.std(list(scores.values()),ddof=1)) if complete else None)
    for a,b in [('view30','baseline'),('view_all','view30'),('view_weighted','view_all'),('view_weighted','view30')]:
        if groups[a]['complete'] and groups[b]['complete']:
            paired={s:100*(groups[a]['seeds'][s]-groups[b]['seeds'][s]) for s in groups[a]['seeds']}
            contrasts[a+'-'+b]=dict(paired_pp=paired,mean_pp=float(np.mean(list(paired.values()))))
    return dict(groups=groups,contrasts=contrasts,results=results,
        note='VOC val development result; synthetic views are not measured 3D rotations; no significance/automatic deployment claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--views-dir',type=Path)
    parser.add_argument('--aug-list',type=Path,required=False)
    parser.add_argument('--train-list',type=Path,default=Path('/home/zystu2025/datasets/VOC2012/ImageSets/Segmentation/trainaug.txt'))
    parser.add_argument('--val-list',type=Path,default=Path('/home/zystu2025/datasets/VOC2012/ImageSets/Segmentation/val.txt'))
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--previous',type=Path,default=ROOT/'experiments/background_partition_resume_20261006_002500')
    parser.add_argument('--version',default='view_aug_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',nargs=2,type=int,default=[0,1])
    parser.add_argument('--max-hours',type=float,default=48)
    parser.add_argument('--extend-hours',type=float,default=0,help='Explicitly add time to an existing resumed version')
    parser.add_argument('--preflight-only',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--validation-only',action='store_true',help='Stop after full baseline replays and real image probes; resume to extract/train')
    opts=parser.parse_args(); os.chdir(ROOT); install_relocation()
    if opts.worker: return worker(ab.read(opts.worker))
    if opts.views_dir is None: parser.error('--views-dir is required (after generation, or for inventory)')
    ab.parse_args(['--version',opts.version])
    if opts.inventory_only:
        report=inventory(opts.views_dir,opts.train_list.read_text().split(),opts.val_list.read_text().split())
        print({k:v for k,v in report.items() if k not in ('files','missing')})
        print('Missing:',len(report['missing']),report['missing'][:20]); return
    if len(set(opts.gpus))!=2 or min(opts.gpus)<0 or not math.isfinite(opts.max_hours) or opts.max_hours<=0:
        raise ValueError('Invalid GPUs/time budget')
    if not math.isfinite(opts.extend_hours) or opts.extend_hours<0 or (opts.extend_hours and not opts.resume):
        raise ValueError('--extend-hours requires --resume and a nonnegative finite value')
    refs,split,records=history(opts.previous.resolve()); common=refs[42]['job']
    inv=inventory(opts.views_dir,common['train_ids'],common['val_ids'],hash_files=True)
    if not inv['complete']: raise ValueError(f'Generation incomplete: {len(inv["missing"])} missing, first={inv["missing"][:10]}')
    listed=verify_aug_list(opts.aug_list,common['train_ids'],inv) if opts.aug_list else None
    selected={k:v for k,v in inv['files'].items() if v['source_id'] in set(split['train_images'])}
    tags={str(r['img_id']):list(dict.fromkeys(int(c) for c in r['labels'] if int(c)!=0))
          for r in ab.read(common['imagelevel_json'])['images']}
    if any(not tags.get(r['source_id']) or not set(tags[r['source_id']])<=set(range(1,21)) for r in selected.values()):
        raise ValueError('Missing/invalid source image tags')
    plan=ROOT/'experiments/mpsa_combo_20260929_105543/members.json'
    if str(plan.resolve()) not in {r['path'] for r in records}: raise ValueError('Original member plan not in trusted history')
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    vocabulary=SAM3PseudoMaskGenerator(dataset_type='voc',load_model=False)
    active_classes=old.canonical([asdict(c) for c in vocabulary.active_classes])
    # Fail here if the Active API changes, before reserving a version or GPU work.
    import inspect
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    inspect.signature(SAM3ActiveProposer).bind(None,None,vocabulary.active_classes,**ACTIVE)
    code=ab.source_identity()
    records += [dict(path=r['path'],size=r['bytes'],sha256=r['sha256']) for r in inv['files'].values()]
    records=unique_records(records)
    if opts.aug_list: records+=ab.identities([opts.aug_list])
    identity=old.canonical(dict(protocol=PROTOCOL,previous=str(opts.previous.resolve()),records=records,code=code,
        views_dir=str(opts.views_dir.resolve()),aug_list=listed,active=ACTIVE,rule=RULE,active_classes=active_classes,
        split=split,gpus=opts.gpus,max_hours=opts.max_hours,generator='user-reported inspatio-world-v1.5'))
    output=ROOT/'experiments'/opts.version; cache=ROOT/'storage/raw_banks/voc'/opts.version/'generated_regions'
    pseudo_root=ROOT/'storage/pseudo_labels/voc'/opts.version/'generated'
    logs=ROOT/'logs'/opts.version; weights=ROOT/'weights/sam3_adapter'/opts.version
    banks=ROOT/'storage/mlp_banks/voc'/opts.version
    if output.exists() and (not opts.resume or ab.read(output/'manifest.json')!=identity):
        raise ValueError('Existing version differs or needs --resume')
    if not output.exists() and any(p.exists() for p in (cache.parent,pseudo_root.parent,logs,weights,banks)):
        raise FileExistsError('Version output paths already used')
    for gpu in opts.gpus: ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    raw_bytes=sum(p.stat().st_size for p in Path(common['raw']).rglob('*.npz'))
    # Conservative initial allowance; runtime floor remains authoritative.
    allowance=12*raw_bytes+len(selected)*200_000+2*2**30
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+allowance:
        raise queue.Deferred(f'Need estimated {allowance/2**30:.1f} GiB plus 5 GiB reserve; no automatic deletion')
    print(f'[Preflight passed] {len(selected)} training-only views; real-only gallery/validation; 9 trainings + 12 full evaluations; HCR off',flush=True)
    if opts.preflight_only: return
    if not output.exists(): ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists(): ab.write(budget,dict(start=time.time(),extensions=[]))
    timing=ab.read(budget)
    if opts.extend_hours:
        timing['extensions'].append(dict(time=time.time(),hours=opts.extend_hours)); ab.write(budget,timing)
    started=timing['start']; deadline=started+(opts.max_hours+sum(r['hours'] for r in timing['extensions']))*3600
    cancel=threading.Event(); guard=threading.Lock(); stages=AliasAwareStages(output,True)
    results={}; failures={}; status='running'; error=None
    entry=queue.__file__; queue.__file__=__file__
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_): cancel.set(); raise KeyboardInterrupt('Interrupted')
    for s in handlers: signal.signal(s,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),failures=failures,expected_evaluations=12))
        ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,**summarize(results)))
    split_path=output/'split.json'
    if not split_path.exists(): ab.write(split_path,split)
    if ab.read(split_path)!=split: raise ValueError('Saved split changed')
    shared=dict(protocol=PROTOCOL,rule=old.canonical(RULE),view_files=selected,tags=tags,split=split,
        view_cache=str(cache),view_pseudo=str(pseudo_root),active_classes=active_classes,source_sha=old.object_hash(code),
        plan=str(plan),probe_id=sorted(split['train_images'])[0])
    def launch(key,job,gpu):
        def action(attempt):
            job.update(shared,attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline)
            if job.get('smoke'):
                job['view_files']={k:selected[k] for k in sorted(selected)[:8]}
            result=dict(attempt=str(attempt)); artifacts=[attempt]
            if job['stage']=='train':
                job.update(weights=str(weights/key/attempt.name),bank=str(banks/key/attempt.name),split_file=str(split_path))
                result.update(weights=job['weights'],bank=job['bank']); artifacts += [Path(job['weights']),Path(job['bank'])]
            env=os.environ.copy(); env.update(refs[job['seed']]['environment']); env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OVERRIDE_SAM3_CKPT=job['checkpoint'])
            env=evaluation_environment(job,env)
            if job['stage']!='eval': env['OVERRIDE_FEATURE_ADAPTER_CKPT']=''
            log=logs/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline); artifacts.append(log)
            if job['stage'] in ('extract','probe'):
                for view_key in job['ids']:
                    marker=cache/view_key/'complete.json'; artifacts += [marker,*[Path(r['path']) for r in ab.read(marker)['artifacts']]]
            return result,artifacts
        return stages.run(key,action)
    def evaluate(method,seed,gpu,trained=None):
        job=copy.deepcopy(refs[seed]['job']); job.update(stage='eval',method=method)
        if trained: job.update(weights=trained['weights'],bank=trained['bank'])
        value=launch(f'eval_{method}_{seed}',job,gpu); attempt=Path(value['attempt'])
        metric=ab.read(attempt/'metrics.json'); ref=ab.read(refs[seed]['metrics'])
        current=provenance.evidence.checked_copy(metric)[0]; reference=provenance.evidence.checked_copy(ref)[0]
        old.validate_eval(current,reference,job['val_ids'],ref['proposal_count_after'],.2)
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
        if method=='baseline':
            old.require_image_replay(refs[seed]['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=ref['confusion_matrix']: raise queue.GlobalFailure('Historical replay differs')
        result=dict(miou=current['mIoU'],metrics=str(attempt/'metrics.json'),
            per_image=str(attempt/'per_image_confusion.npz'),per_class_iou=current['per_class_iou'],
            worker=ab.read(attempt/'queue_worker.json'),training=trained)
        with guard: results[f'{method}_{seed}']=result; flush()
    def fatal(_,exc): raise exc
    try:
        flush(); stages.verify_all()
        queue.run_round(list(old.SEEDS),opts.gpus,lambda s,g:evaluate('baseline',s,g),cancel,deadline,fatal)
        ids=sorted(selected)
        launch('probe',dict(common,stage='probe',ids=ids[:8]),opts.gpus[0])
        smoke=launch('train_smoke',dict(common,stage='train',method='view_weighted',smoke=True),opts.gpus[0])
        smoke_eval=launch('eval_smoke',dict(common,stage='eval',method='view_weighted',smoke=True,
            weights=smoke['weights'],bank=smoke['bank']),opts.gpus[0])
        smoke_metrics=ab.read(Path(smoke_eval['attempt'])/'metrics.json')
        old.seeds.validate_image_matrices(str(Path(smoke_eval['attempt'])/'per_image_confusion.npz'),smoke_metrics)
        if opts.validation_only:
            status='validated'; return
        queue.run_round([0,1],opts.gpus,lambda lane,g:launch(f'extract_{lane}',dict(common,stage='extract',ids=ids[lane::2]),g),cancel,deadline,fatal)
        def task(item,gpu):
            method,seed=item
            trained=launch(f'train_{method}_{seed}',dict(refs[seed]['job'],stage='train',method=method),gpu)
            evaluate(method,seed,gpu,trained)
        def failed(item,exc):
            with guard: failures[str(item)]=str(exc); flush()
        queue.run_round([(m,s) for s in old.SEEDS for m in METHODS[1:]],opts.gpus,task,cancel,deadline,failed)
        gmp.verify(records,'Inputs after execution'); gmp.verify(code,'Frozen code'); stages.verify_all()
        status='complete' if len(results)==12 else 'partial'
    except queue.Deferred as exc: status='deferred'; error=str(exc)
    except BaseException as exc: status='failed'; error=str(exc); cancel.set(); raise
    finally:
        flush(); ab.write(output/'artifacts.json',dict(owned_roots=[str(p) for p in (output,cache.parent,pseudo_root.parent,logs,weights,banks)],automatic_deletion=False))
        queue.__file__=entry
        for s,h in handlers.items(): signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(opts.version,status,str(output),time.time()-started)


if __name__=='__main__': main()
