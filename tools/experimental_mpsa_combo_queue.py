#!/usr/bin/env python3
"""Fixed 48-hour dual-GPU MPSA x pooling queue; no production modifications."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_pmfa_ab as ab
from tools import experimental_rpg_ab as rpg
from tools import experimental_gmp_ab as gmp
from tools import experimental_gmp_seeds as seeds
from tools import experimental_mpsa_ab as mpsa

GROUPS = dict(A=('mean', 1), B=('mean', 2), C=('gmp', 1), D=('gmp', 2), E=('gate', 1), F=('gate', 2))
SEEDS = (42, 123, 2026)
FORMAT = 3006
FLOOR = 5 * 2**30
KEY_IMAGES = {'2009_000732', '2010_004472', '2009_002346', '2009_003387', '2007_007417'}
OLD_PMFA = ab.PMFA
OLD_SIGNATURE = ab.signature


class GlobalFailure(RuntimeError): pass
class Deferred(RuntimeError): pass
class TaskFailure(RuntimeError): pass
class TaskOOM(TaskFailure): pass


@dataclass(frozen=True)
class Recipe:
    group: str
    checkpoint_sha256: str

    def __post_init__(self):
        if self.group not in {*GROUPS, 'SMFA'}: raise ValueError('Unknown method')
        mpsa.MPSA(1, self.checkpoint_sha256)


def signature(base, mode, recipe):
    if recipe.group == 'SMFA':
        result = OLD_SIGNATURE(base, 'smfa', OLD_PMFA())
        result['format_version'] = 3001
        return result
    method, count = GROUPS[recipe.group]
    if method == 'mean': return mpsa.signature(base, 'mpsa', mpsa.MPSA(count, recipe.checkpoint_sha256))
    return dict(base, format_version=FORMAT, experimental_pooling=dict(
        mode='mpsa_combo', version=1, parameters=asdict(recipe), phases=count,
        offsets=[[0, 0]] if count == 1 else [[0, 0], [7, 7]], geometry=mpsa.signature({}, 'mpsa', mpsa.MPSA(count, recipe.checkpoint_sha256))['experimental_pooling'],
        inner=method, gmp_rho=.1 if method == 'gmp' else None,
        gmp_weights='uniform_on_hard_selected_tokens' if method == 'gmp' else None,
        gate=dict(floor=.5, temperature=.1, median='torch_lower', proxy='normalized_raw_hard_mean', zero_proxy='hard_mean') if method == 'gate' else None,
        precision_policy=ab.POLICY, placement='before_region_mlp'))


def gate_pool(tokens):
    import torch
    import torch.nn.functional as F
    if tokens.ndim != 2 or not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError('Invalid gate tokens')
    if len(tokens) == 0: return tokens.new_zeros(tokens.shape[1]), dict(empty=True)
    mean = tokens.float().mean(0)
    if mean.norm() <= 1e-12: return F.normalize(mean, dim=0), dict(zero_proxy=True)
    sim = F.normalize(tokens.float(), dim=1) @ F.normalize(mean, dim=0)
    weights = .5 + .5 * torch.sigmoid((sim-sim.median())/.1)
    result = F.normalize((tokens.float()*weights[:, None]).sum(0)/weights.sum(), dim=0)
    return result, dict(gate_min=float(weights.min()), gate_max=float(weights.max()))


def extract(model, rgb, masks, include_smfa=False, diagnostics=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    if model.resolution != 1008 or model.context_subtract != 0 or model.use_bf16:
        raise ValueError('Expected original MPSA geometry/FP32')
    keys = list(GROUPS) + (['SMFA'] if include_smfa else [])
    if not masks: return {k: np.zeros((0, model.embed_dim), np.float32) for k in keys}
    if any(np.asarray(mask).shape != rgb.shape[:2] for mask in masks): raise ValueError('Mask/image mismatch')
    inp, transform = model._preprocess_image(rgb)
    phase_values = {}; baseline = None
    with torch.inference_mode():
        for phase, offset in enumerate(((0, 0), (7, 7))):
            tokens, grid = model._forward_tokens(mpsa.shift_input(inp, offset))
            if tuple(grid) != (72, 72) or tokens.shape != (1, 5184, model.embed_dim): raise ValueError('Grid changed')
            by_method = {k: [] for k in ('mean', 'gmp', 'gate')}
            for start in range(0, len(masks), 16):
                for i in range(start, min(start+16, len(masks))):
                    index, info = mpsa.selection(masks[i], offset)
                    selected = tokens[0, torch.as_tensor(index, device=tokens.device, dtype=torch.long)].float()
                    mean = mpsa.hard_mean(tokens[0], index)
                    value, solve = gmp.solve(selected, torch.ones(len(selected), device=tokens.device), .1)
                    gate, gate_info = gate_pool(selected)
                    for name, vec in (('mean', mean), ('gmp', value), ('gate', gate)): by_method[name].append(vec)
                    if diagnostics is not None: diagnostics.append(dict(region=i, phase=phase, **info, gmp=solve, gate=gate_info))
            if include_smfa and phase == 0:
                # Preserve original Base.extract_features_batch matrix-multiply
                # order for the reference only; no coverage enters A--F.
                coverages=[model._downsample_mask_to_patches(np.asarray(mask,bool),72,72,transform).astype(np.float32) for mask in masks]
                valid=[i for i,w in enumerate(coverages) if w.sum()>=1e-6]
                baseline=tokens.new_zeros((len(masks),model.embed_dim))
                if valid:
                    w=torch.from_numpy(np.vstack([coverages[i] for i in valid])).to(device=tokens.device,dtype=tokens.dtype)
                    baseline[valid]=F.normalize(torch.matmul(w,tokens[0])/(w.sum(1,keepdim=True)+1e-8),dim=1)
            phase_values[phase] = {k: torch.stack(v) for k, v in by_method.items()}
            del tokens
        output = {}
        for group, (method, count) in GROUPS.items():
            # Match the old MPSA stack/mean/normalize operation order exactly.
            output[group] = F.normalize(torch.stack([phase_values[p][method] for p in range(count)]).mean(0), dim=1).cpu().numpy()
        if include_smfa: output['SMFA'] = baseline.cpu().numpy()
    return output


def mask_hashes(masks):
    import numpy as np
    return [hashlib.sha256(np.asarray(m, dtype=np.uint8).tobytes()).hexdigest() for m in masks]


def ensure_resources(deadline, root=ROOT):
    if time.time() >= deadline: raise Deferred('Run time budget reached; increase --max-hours explicitly to extend')
    if shutil.disk_usage(root).free < FLOOR: raise Deferred('Less than 5 GiB free; retained all artifacts')


def cache_read(folder, identity=None):
    marker = Path(folder)/'complete.json'
    if not marker.exists(): return None
    value = ab.read(marker)
    try: gmp.verify(value['artifacts'], 'Shared cache')
    except Exception as exc: raise GlobalFailure(str(exc)) from exc
    if identity is not None and value['identity'] != identity: raise GlobalFailure('Shared cache identity mismatch')
    return value


def cache_vectors(folder, identity, group, hashes):
    import numpy as np
    value = cache_read(folder, identity)
    if value is None: raise GlobalFailure('Missing committed query cache')
    if value['mask_hashes'] != hashes: raise GlobalFailure('Query mask order/content changed')
    with np.load(value['file'], allow_pickle=False) as data: result = data[group].copy()
    if result.shape != (len(hashes), 1280) or not np.isfinite(result).all(): raise GlobalFailure('Invalid query vectors')
    return result


def next_attempt(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while (folder/f'attempt_{n:03d}').exists(): n += 1
    target = folder/f'attempt_{n:03d}'; target.mkdir(); return target


def extract_worker(job):
    import numpy as np
    import torch
    from PIL import Image
    from src.feature_extractor_factory import build_feature_extractor
    from src.region_extractors import get_regions_from_pseudo
    model = build_feature_extractor(torch.device('cuda:0'), use_interactivity=False)
    source = model.feature_signature()
    if source != job['source']: raise GlobalFailure('Live SAM3 source signature differs')
    references = {k: gmp.read_bank(job['references'][k]['raw']) for k in ('A', 'B')}
    plan = ab.read(job['plan']); diagnostics_count = Counter()
    for stem in job['ids']:
        ensure_resources(job['deadline'])
        training = stem in plan
        folder = Path(job['cache'])/('train' if training else 'val')/stem
        identity = job['cache_identity'][stem]
        if cache_read(folder, identity): continue
        mask_source=Path(job['pseudo'])/f'{stem}.png' if training else Path(job['proposals'])/f'{stem}.npz'
        if ab.digest(job['images'][stem])!=identity['image_sha256'] or ab.digest(mask_source)!=identity['mask_source_sha256']:
            raise GlobalFailure('Extraction input changed')
        with Image.open(job['images'][stem]) as im: rgb = np.asarray(im.convert('RGB'))
        if training:
            regions = get_regions_from_pseudo(job['images'][stem], 300, job['pseudo'])
            rows = plan[stem]; masks = [regions[r['index']]['mask'] for r in rows]
            if [hashlib.sha256(x.tobytes()).hexdigest() for x in masks] != [r['mask_sha256'] for r in rows]:
                raise GlobalFailure('Fixed training masks changed')
        else:
            rows = []
            with np.load(Path(job['proposals'])/f'{stem}.npz', allow_pickle=False) as data: masks = list(data['masks'].astype(bool))
        start = time.monotonic(); diagnostics = []
        vectors = extract(model, rgb, masks, not training, diagnostics)
        if training:
            for group in ('A', 'B'):
                old = np.asarray([references[group][(r['cls'], r['id'])] for r in rows], np.float32).reshape(-1,1280)
                if not np.allclose(vectors[group], old, atol=5e-5, rtol=5e-5): raise GlobalFailure(f'Old {group} raw vectors not reproduced: {stem}')
            if any((np.linalg.norm(v, axis=1)<1e-6).any() for v in vectors.values()): raise GlobalFailure('Zero retained training vector')
        # Probe compares shared-phase code to the original MPSA implementation.
        if job['stage'] == 'probe' and masks:
            original = mpsa.extract_many(model, rgb, masks[:2], (1,2))
            for group, count in (('A',1),('B',2)):
                if not np.allclose(vectors[group][:2], np.stack([v.cpu().numpy() for v in original[count]]), atol=5e-5, rtol=5e-5):
                    raise GlobalFailure('Online/shared MPSA mismatch')
            single = extract(model, rgb, masks[:1], not training)
            if any(not np.allclose(vectors[k][:1], single[k], atol=5e-5, rtol=5e-5) for k in vectors):
                raise GlobalFailure('Single/batch combo mismatch')
        attempt = next_attempt(folder); file = attempt/'vectors.npz'
        saved = {k:v for k,v in vectors.items() if not training or k in 'CDEF'}
        np.savez_compressed(file, **saved, ids=np.asarray([r['id'] for r in rows], dtype=str), classes=np.asarray([r['cls'] for r in rows], dtype=str))
        ab.write(attempt/'diagnostics.json', dict(image=stem, seconds=time.monotonic()-start, regions=diagnostics))
        ab.write(folder/'complete.json', dict(identity=identity, mask_hashes=mask_hashes(masks), file=str(file),
            artifacts=ab.identities([file, attempt/'diagnostics.json'])))
        # Serialization must preserve exact vectors; this also exercises query checks.
        if not training:
            for group in vectors:
                if not np.array_equal(cache_vectors(folder, identity, group, mask_hashes(masks)), vectors[group]):
                    raise GlobalFailure('Cache/online vectors differ')
        diagnostics_count['images'] += 1; diagnostics_count['regions'] += len(masks)
        diagnostics_count['nearest_fallbacks'] += sum(int(r['nearest_fallback']) for r in diagnostics)
        print(f'[Cache {job["stage"]}] {stem}: {len(masks)} regions', flush=True)
    return dict(pooling_stats=dict(diagnostics_count))


def install_cached_eval(job):
    import numpy as np
    import torch
    import src.feature_signature as fs
    from src.feature_adapter import load_adapter, project_features, adapted_signature
    fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
    mpsa.install_clean_eval(job)
    import run_inference as inference
    import src.inference as inf
    state = {}; source_sig = job['raw_signature']
    class CachedExtractor:
        def __init__(self, device):
            self.device = device
            self.adapter, _ = load_adapter(Path(job['weights'])/'adapter.pt', device, source_sig)
            self.embed_dim = self.adapter.architecture['output_dim']
        def feature_signature(self):
            return adapted_signature(source_sig, Path(job['weights'])/'adapter.pt', self.adapter.architecture)
        def extract_features_batch(self, rgb, masks):
            stem = state['id']
            if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() != state['rgb_hash']:
                raise GlobalFailure('Query image pixels changed')
            raw = cache_vectors(Path(job['cache'])/'val'/stem, job['cache_identity'][stem], job['group'], mask_hashes(masks))
            projected=project_features(self.adapter, raw, self.device)
            state['valid_indices']=np.flatnonzero(np.isfinite(projected).all(1)&(np.linalg.norm(projected,axis=1)>=1e-6)).tolist()
            return list(projected)
    inference.build_feature_extractor = lambda device, **_: CachedExtractor(device)
    original_load = inference.load_strict_proposals
    def load(directory, stem, height, width):
        from PIL import Image
        state.clear(); state['id'] = stem
        if ab.digest(job['images'][stem]) != job['cache_identity'][stem]['image_sha256']: raise GlobalFailure('Query image file changed')
        if ab.digest(Path(directory)/f'{stem}.npz') != job['cache_identity'][stem]['mask_source_sha256']: raise GlobalFailure('Query proposal file changed')
        with Image.open(job['images'][stem]) as im: state['rgb_hash'] = hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
        masks, scores = original_load(directory, stem, height, width)
        state['masks'] = list(masks); state['hashes'] = mask_hashes(masks)
        state['scores'] = list(map(float, scores)); state['retrieval'] = []; state['before_nms'] = []; state['after_nms'] = []
        return masks, scores
    inference.load_strict_proposals = load
    classify = inf.classify_queries
    def classify_record(*args, **kwargs):
        result = classify(*args, **kwargs)
        if state.get('id') in KEY_IMAGES:
            state['retrieval'] = [dict(proposal_index=i,mask_sha256=state['hashes'][i],
                passes_margin=r.margin>=inf.INF_RETRIEVAL_MARGIN_THR,**vars(r)) for i,r in zip(state['valid_indices'],result)]
        return result
    inf.classify_queries = classify_record
    nms = inf.nms_masks_per_class
    def simple(rows):
        return [dict(mask_sha256=mask_hashes([r['mask']])[0], **{k:v for k,v in r.items() if k!='mask'}) for r in rows]
    def record_nms(rows, threshold):
        result = nms(rows, threshold)
        if state.get('id') in KEY_IMAGES: state['before_nms'], state['after_nms'] = simple(rows), simple(result)
        return result
    inf.nms_masks_per_class = record_nms
    rasterize = inference.rasterize_instances
    def record_rasterize(rows, height, width):
        result = rasterize(rows, height, width)
        if state.get('id') in KEY_IMAGES:
            folder = Path(job['attempt'])/'key_images'; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/(state['id']+'.npz'), prediction=result)
            seen = np.zeros((height,width), bool); composition = []
            for row in rows:
                participates = row['cls_name'] != 'background' and row['cls_name'] in inf.DATASET_CLASSES
                assigned = row['mask'] & ~seen if participates else np.zeros_like(seen)
                composition.append(dict(mask_sha256=mask_hashes([row['mask']])[0], cls_name=row['cls_name'], assigned_pixels=int(assigned.sum())))
                seen |= assigned
            ab.write(folder/(state['id']+'.json'), dict(image_id=state['id'], proposal_hashes=state['hashes'],
                proposal_scores=state['scores'], retrieval=state['retrieval'], before_nms=state['before_nms'],
                after_nms=state['after_nms'], composition=composition))
        return result
    inference.rasterize_instances = record_rasterize


def worker(job):
    os.chdir(ROOT); sys.path.insert(0, str(ROOT/'sam3-main'))
    started = time.monotonic(); result = {}; status = 'failed'; category = 'local'
    import torch
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    try:
        if job['stage'] in ('extract','probe'):
            ab.install_precision()
            import configs.config as cfg
            cfg.DATASET_ROOT = cfg.INF_DATASET_ROOT = job['dataset_root']
            from project_utils.seed_utils import set_global_seed
            set_global_seed(42)
            result = extract_worker(job)
        else:
            ab.PMFA = Recipe; ab.signature = signature
            ab.EXPERIMENT_FORMAT = job['raw_signature']['format_version']
            job['mode'] = 'combo'; job['pmfa'] = asdict(Recipe(job['group'],job['checkpoint_sha256']))
            def install(*_):
                import src.feature_signature as fs
                fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
                if job['stage'] == 'eval': install_cached_eval(job)
            ab.install_pooling = install
            if job['stage'] == 'train':
                original = ab.runpy.run_path
                def train(path, *args, **kwargs):
                    if Path(path).resolve() != ROOT/'tools/train_feature_adapter.py': raise GlobalFailure('Unexpected trainer')
                    sys.argv = seeds.training_argv(sys.argv, job['seed'])
                    ab.write(Path(job['attempt'])/'effective_training_argv.json',dict(argv=sys.argv,seed=job['seed']))
                    return original(path,*args,**kwargs)
                ab.runpy.run_path = train
            ab.worker(job)
        torch.cuda.synchronize(); status = 'complete'
    except Deferred as exc:
        category='deferred'; result['error']=str(exc); raise
    except torch.cuda.OutOfMemoryError as exc:
        category='oom'; result['error']=str(exc); raise
    except (GlobalFailure,ValueError,KeyError,FileNotFoundError) as exc:
        category='global'; result['error']=str(exc); raise
    except BaseException as exc:
        result['error']=str(exc); raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,category=category,
            seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),precision_policy=ab.POLICY,**result))


def verified_stage(folder, key, records):
    value=ab.read(Path(folder)/key/'complete.json')
    gmp.verify(value['artifacts'], key); records.extend(value['artifacts'])
    return {k:v for k,v in value['result'].items() if k!='worker_report'}


def history(mpsa_previous, seed_previous):
    manifest=ab.read(mpsa_previous/'manifest.json'); seed_manifest=ab.read(seed_previous/'manifest.json')
    for folder, data in ((mpsa_previous,manifest),(seed_previous,seed_manifest)):
        if ab.read(folder/'status.json')['status']!='complete': raise GlobalFailure('Historical experiment incomplete')
        gmp.verify(data['code'],'Historical source'); gmp.verify(data['inputs'],'Historical inputs')
    rpg.verify_records=gmp.verify
    old,base,records=mpsa.checked_baseline(Path(manifest['previous']))
    records+=ab.identities([p/n for p in (mpsa_previous,seed_previous) for n in ('manifest.json','status.json','summary.json')])
    # Verify the seed run used the identical historical SMFA source.
    gmp_manifest=ab.read(Path(seed_manifest['previous'])/'manifest.json')
    if Path(gmp_manifest['previous']).resolve()!=Path(manifest['previous']).resolve(): raise GlobalFailure('Different SMFA baselines')
    refs={}; smfa={42:base}
    for group,count in (('A',1),('B',2)):
        refs[group]={}
        for key in (f'merge_{count}',f'train_{count}',f'eval_{count}'):
            refs[group].update(verified_stage(mpsa_previous,key,records))
    for seed in (123,2026):
        smfa[seed]=dict(raw=base['raw'])
        for kind in ('train','eval'): smfa[seed].update(verified_stage(seed_previous,f'smfa_seed{seed}_{kind}',records))
    split=ab.read(Path(base['weights'])/'training_report.json')
    for value in [*refs.values(),*smfa.values()]:
        training=ab.read(Path(value['weights'])/'training_report.json')
        if any(training[k]!=split[k] for k in ('train_images','validation_images')): raise GlobalFailure('Historical splits differ')
        if ab.bank_members(value['raw'])!=ab.bank_members(base['raw']): raise GlobalFailure('Historical raw members differ')
    return old,base,refs,smfa,records


def launch(job, attempt, env, log, cancel, deadline, respect_deadline=True):
    # Separate launcher avoids altering shared global check_gpu in two threads.
    while True:
        if cancel.is_set(): raise GlobalFailure('Queue cancelled')
        ensure_resources(deadline if respect_deadline else float('inf'))
        try:
            line=subprocess.check_output(['nvidia-smi',f'--id={env["CUDA_VISIBLE_DEVICES"]}',
                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used,util=map(float,line.strip().split(','))
            if not all(math.isfinite(v) and v>=0 for v in (used,util)): raise GlobalFailure('Invalid GPU telemetry')
            if used<=1000 and util<=10: break
        except subprocess.CalledProcessError as exc: raise GlobalFailure('GPU telemetry unavailable') from exc
        cancel.wait(10)
    ab.write(attempt/'job.json',job)
    cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker',str(attempt/'job.json')]
    ab.write(attempt/'command.json',dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('OVERRIDE_') or k=='CUDA_VISIBLE_DEVICES'}))
    log.parent.mkdir(parents=True,exist_ok=True); process=None
    try:
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            print(f'[Worker] {job["stage"]} pid={process.pid} GPU={env["CUDA_VISIBLE_DEVICES"]} log={log}',flush=True)
            while process.poll() is None:
                if cancel.wait(1): raise GlobalFailure('Queue cancelled')
            if process.returncode:
                report=ab.read(attempt/'queue_worker.json') if (attempt/'queue_worker.json').exists() else {}
                kind=report.get('category'); error=report.get('error',f'Worker exit {process.returncode}; {log}')
                cls={'global':GlobalFailure,'deferred':Deferred,'oom':TaskOOM}.get(kind,TaskFailure)
                raise cls(error)
    finally:
        if process and process.poll() is None:
            if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
            else: process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
                else: process.kill()
                process.wait()


def task_order(seed):
    return [(g,seed) for g in GROUPS if seed!=42 or g not in 'AB']


def run_round(items, gpus, action, cancel, deadline, on_failure):
    pending=list(items); lock=threading.Lock(); deferred=[]
    def lane(gpu):
        while not cancel.is_set():
            with lock:
                if not pending: return
                try: ensure_resources(deadline)
                except Deferred as exc: deferred.append(str(exc)); return
                item=pending.pop(0)
            try: action(item,gpu)
            except Deferred as exc:
                with lock: deferred.append(str(exc))
                return
            except TaskFailure as exc:
                try: on_failure(item,exc)
                except BaseException:
                    cancel.set(); raise
            except BaseException:
                cancel.set(); raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(lane,gpu) for gpu in gpus]
        for future in futures: future.result()
    if deferred: raise Deferred(deferred[0])


def merge(cache, ids, raw, base_raw, group, sig, identities):
    import numpy as np
    from src.feature_bank_storage import save_feature_matrix_npz
    wanted=gmp.read_bank(base_raw); rows={}
    for stem in ids:
        item=cache_read(Path(cache)/'train'/stem,identities[stem])
        if item is None: raise GlobalFailure('Missing training cache')
        with np.load(item['file'],allow_pickle=False) as data:
            if len(data[group])!=len(data['ids']): raise GlobalFailure('Cache rows differ')
            for cls,sid,vec in zip(data['classes'],data['ids'],data[group]):
                key=(str(cls),str(sid))
                if key in rows: raise GlobalFailure('Duplicate member')
                rows[key]=vec
    if set(rows)!=set(wanted): raise GlobalFailure('Members changed')
    raw.mkdir(parents=True,exist_ok=False)
    for cls in sorted({k[0] for k in rows}):
        keys=sorted(k for k in rows if k[0]==cls)
        save_feature_matrix_npz(raw/'features'/cls/'features.npz',np.stack([rows[k] for k in keys]),[k[1] for k in keys])
    ab.write(raw/'meta/feature_signature.json',sig)
    shutil.copy2(Path(base_raw)/'meta/class_mapping.json',raw/'meta/class_mapping.json')


def compare(reference,current,ids):
    value=gmp.report(reference,current,ids)
    value['candidate_miou']=value.pop('gmp_miou')
    value['confusion_matrices']['candidate']=value['confusion_matrices'].pop('gmp')
    for row in value['per_class_iou'].values(): row['candidate']=row.pop('gmp')
    value['note']='Predeclared paired experiment; no validation-driven search or significance claim.'
    return value


def summarize(reports):
    import statistics
    by_group={}; contrasts={}
    for group in GROUPS:
        rows=[dict(seed=seed,**reports[f'{group}_{seed}']) for seed in SEEDS if f'{group}_{seed}' in reports]
        deltas=[r['delta_pp'] for r in rows]
        by_group[group]=dict(complete=len(rows)==3,completed_seeds=[r['seed'] for r in rows],
            mean_delta_pp=statistics.mean(deltas) if deltas else None,
            sample_std_delta_pp=statistics.stdev(deltas) if len(deltas)>1 else None)
    for seed in SEEDS:
        def delta(a,b):
            return 100*(reports[f'{a}_{seed}']['candidate_miou']-reports[f'{b}_{seed}']['candidate_miou']) if all(f'{g}_{seed}' in reports for g in (a,b)) else None
        values={f'{a}-{b}':delta(a,b) for a,b in [('C','A'),('D','B'),('E','A'),('F','B')]}
        for name,one,two in [('gmp_interaction','D-B','C-A'),('gate_interaction','F-B','E-A')]:
            values[name]=values[one]-values[two] if values[one] is not None and values[two] is not None else None
        contrasts[str(seed)]=values
    return dict(groups=by_group,contrasts_pp=contrasts,results=reports)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--version',default='mpsa_combo_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--gpus',nargs=2,type=int,default=[0,1]); parser.add_argument('--max-hours',type=float,default=48)
    parser.add_argument('--mpsa-previous',type=Path,default=Path('experiments/mpsa_20260928_172820'))
    parser.add_argument('--smfa-seeds-previous',type=Path,default=Path('experiments/gmp_seeds_20260927_194351'))
    parser.add_argument('--clean-proposals-dir',type=Path,default=Path(mpsa.DEFAULT_CLEAN))
    parser.add_argument('--preflight-only',action='store_true'); parser.add_argument('--resume',action='store_true')
    opts=parser.parse_args(argv); os.chdir(ROOT)
    if opts.worker: return worker(ab.read(opts.worker))
    if len(set(opts.gpus))!=2 or min(opts.gpus)<0 or not math.isfinite(opts.max_hours) or opts.max_hours<=0: parser.error('Invalid GPUs or time budget')
    ab.parse_args(['--version',opts.version])
    old,base,refs,smfa,records=history(opts.mpsa_previous.resolve(),opts.smfa_seeds_previous.resolve())
    flags=['--version',opts.version,'--gpu',str(opts.gpus[0])]
    for key in ('dataset_root','image_dir','checkpoint','train_list','val_list','imagelevel_json','gt_dir'): flags+=['--'+key.replace('_','-'),old['options'][key]]
    old_folder=Path(ab.read(opts.mpsa_previous/'manifest.json')['previous'])
    pseudo=old['selected'].get('pseudo') or ab.read(old_folder/'pseudo/complete.json')['result']['pseudo']
    proposals=old['selected'].get('proposals') or ab.read(old_folder/'proposals/complete.json')['result']['proposals']
    args=ab.parse_args(flags+['--pseudo-dir',pseudo,'--proposals-dir',str(opts.clean_proposals_dir.resolve())])
    context=ab.preflight(args); ab.check_gpu(opts.gpus[1])
    if not all(context['selected'].values()): raise GlobalFailure('Complete existing inputs required')
    clean=mpsa.validate_clean_cache(args.proposals_dir,context['val_ids'],proposals,ab.read(base['metrics'])['proposal_count_after'])
    context['inputs']+=ab.identities([args.proposals_dir/'cleanup_config.json'])
    source=ab.read(Path(base['raw'])/'meta/feature_signature.json').copy()
    source.pop('experimental_pooling'); source['format_version']=2
    checkpoint=ab.digest(args.checkpoint)
    signatures={g:signature(source,'combo',Recipe(g,checkpoint)) for g in [*GROUPS,'SMFA']}
    for group in ('A','B'):
        if signatures[group]!=ab.read(Path(refs[group]['raw'])/'meta/feature_signature.json'): raise GlobalFailure('Historical MPSA signature differs')
    if signatures['SMFA']!=ab.read(Path(base['raw'])/'meta/feature_signature.json'): raise GlobalFailure('Historical SMFA signature differs')
    output=ROOT/'experiments'/opts.version; raw_root=ROOT/'storage/raw_banks/voc'/opts.version
    cache=raw_root/'region_cache'; weights=ROOT/'weights/sam3_adapter'/opts.version
    banks=ROOT/'storage/mlp_banks/voc'/opts.version; logs=ROOT/'logs'/opts.version
    identity=dict(protocol='mpsa_combo_queue_v1',code=context['code'],inputs=context['inputs'],references=records,
        signatures=signatures,gpus=opts.gpus,clean=clean,seeds=SEEDS)
    # JSON round-trip keeps tuple/list equality stable on resume.
    identity=json.loads(json.dumps(identity))
    if output.exists():
        if not opts.resume or ab.read(output/'manifest.json')!=identity: raise GlobalFailure('Version differs or --resume missing')
    elif any(p.exists() for p in (raw_root,weights,banks,logs)): raise FileExistsError('Version already used')
    members=sum(len(v) for v in ab.bank_members(base['raw']).values())
    # Two copies of four raw banks, sixteen projected banks, query caches, checkpoints/log headroom.
    estimate=members*1280*4*8 + members*256*4*16 + clean['count']*1280*4*7 + 2*2**30
    if shutil.disk_usage(ROOT).free < FLOOR+(0 if opts.resume else estimate): raise Deferred(f'Insufficient disk: estimated new bytes={estimate}, plus 5 GiB reserve')
    if opts.preflight_only:
        print(f'[Preflight passed] 16 new MLP jobs; no recleaning; estimated additional {estimate/2**30:.2f} GiB + 5 GiB reserve.',flush=True); return
    if not output.exists(): ab.write(output/'manifest.json',identity)
    ab.Stages(output,True).verify_all()
    lock=output/'launcher.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists(): ab.write(budget,dict(started_at=time.time()))
    deadline=ab.read(budget)['started_at']+opts.max_hours*3600
    ab.write(output/'effective_budget.json',dict(deadline=deadline,max_hours=opts.max_hours))
    cancel=threading.Event(); previous_signal=signal.getsignal(signal.SIGTERM)
    def stop(*_): cancel.set(); raise KeyboardInterrupt('Queue stopped by user')
    signal.signal(signal.SIGTERM,stop)
    stages=ab.Stages(output,True); report_lock=threading.Lock()
    reports={p.parent.name.removeprefix('report_'):ab.read(p)['result'] for p in output.glob('report_*/complete.json')}
    failures={}; status='incomplete'; error=None
    split=ab.read(Path(base['weights'])/'training_report.json')
    input_hash={r['path']:r['sha256'] for r in context['inputs']}
    cache_identity={}; train_set=set(context['train_ids'])
    space_hash=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
    for stem in context['train_ids']+context['val_ids']:
        mask_source=Path(pseudo)/f'{stem}.png' if stem in train_set else args.proposals_dir/f'{stem}.npz'
        cache_identity[stem]=dict(image_sha256=input_hash[str(Path(context['images'][stem]).resolve())],
            mask_source_sha256=input_hash[str(mask_source.resolve())],feature_space_sha256=space_hash)
    common={k:str(getattr(args,k)) for k in ('dataset_root','image_dir','checkpoint','train_list','val_list','imagelevel_json','gt_dir')}
    common.update(images=context['images'],train_ids=context['train_ids'],val_ids=context['val_ids'],source=source,
        cache=str(cache),cache_identity=cache_identity,checkpoint_sha256=checkpoint,pseudo=pseudo,proposals=str(args.proposals_dir),
        references=refs,plan=str(output/'members.json'),deadline=deadline)
    def flush():
        with report_lock:
            workers={p.parent.name:ab.read(ab.read(p)['result']['queue_worker']) for p in output.glob('*/complete.json') if 'queue_worker' in ab.read(p)['result']}
            ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,workers=workers,
                deadline=deadline,proposal_policy=clean,**summarize(reports)))
    def execute(key,kind,gpu,group='D',seed=42,result=None,ids=None,accepted=False):
        result=result or {}
        def action(attempt):
            job=dict(common,attempt=str(attempt),stage=kind,group=group,seed=seed,raw_signature=signatures[group])
            if ids is not None: job['ids']=ids
            env=ab.base_env(gpu); env['OVERRIDE_SAM3_CKPT']=str(args.checkpoint)
            values={}; artifacts=[attempt]
            if kind=='train':
                w=weights/f'{group}_{seed}'/attempt.name; b=banks/f'{group}_{seed}'/attempt.name
                job.update(raw=result['raw'],weights=str(w),bank=str(b),split_file=str(Path(base['weights'])/'training_report.json'))
                values.update(weights=str(w),bank=str(b)); artifacts += [w,b]
            if kind=='eval':
                job.update(weights=result['weights'],bank=result['bank'],metrics=str(attempt/'metrics.json'),cleanup='PRE_CLEANED')
                env.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(result['weights'])/'adapter.pt'),OVERRIDE_INF_FEATURE_BANK_ROOT=result['bank'],OVERRIDE_INF_PROPOSALS_DIR=str(args.proposals_dir))
                values.update(metrics=job['metrics'],per_image=str(attempt/'per_image_confusion.npz'))
            log=logs/f'{key}_{attempt.name}.log'
            launch(job,attempt,env,log,cancel,deadline,respect_deadline=not accepted)
            if ab.read(attempt/'queue_worker.json')['status']!='complete': raise GlobalFailure('Worker did not commit success')
            if kind=='train':
                training=ab.read(Path(values['weights'])/'training_report.json')
                if any(training[k]!=split[k] for k in ('train_images','validation_images')): raise GlobalFailure('Training split changed')
                if ab.read(attempt/'effective_training_argv.json')['seed']!=seed: raise GlobalFailure('Training seed changed')
            if kind=='eval':
                metrics=ab.read(values['metrics']); seeds.validate_image_matrices(values['per_image'],metrics)
                if metrics['cleanup'] is not None or metrics['proposal_count_before']!=clean['count'] or metrics['proposal_count_after']!=clean['count']: raise GlobalFailure('Unexpected proposal cleaning/count')
            values['queue_worker']=str(attempt/'queue_worker.json')
            return values,artifacts+[log]
        try:
            value=stages.run(key,action)
            with report_lock: ab.write(output/'progress.json',dict(last_completed=key,updated_at=time.time(),deadline=deadline))
            return value
        except (ValueError,KeyError,FileNotFoundError) as exc: raise GlobalFailure(str(exc)) from exc
    def fail(item,exc):
        with report_lock: failures[f'{item[0]}_{item[1]}']=dict(error=str(exc),type=type(exc).__name__)
        flush()
    def commit_report(key,value):
        def action(attempt):
            file=output/f'comparison_{key}.json'; ab.write(file,value)
            return value,[file]
        result=stages.run('report_'+key,action)
        with report_lock: reports[key]=result
    try:
        ab.write(output/'status.json',dict(status='running',completed=len(reports),expected=18,deadline=deadline))
        ab.write(output/'artifacts.json',dict(owned_roots=list(map(str,[output,raw_root,weights,banks,logs])),read_only_inputs=context['selected'],automatic_deletion=False))
        def membership(attempt):
            for k,v in ab.base_env(opts.gpus[0]).items():
                if k.startswith('OVERRIDE_'): os.environ[k]=v
            plan=gmp.enumerate_members(context['train_ids'],context['images'],pseudo,base['raw'])
            ab.write(output/'members.json',plan); return {},[output/'members.json']
        ensure_resources(deadline); stages.run('membership',membership)
        probe_ids=random.Random(42).sample(sorted(split['train_images']),min(4,len(split['train_images'])))+context['val_ids'][:2]
        execute('probe','probe',opts.gpus[0],ids=probe_ids)
        all_ids=context['train_ids']+context['val_ids']
        run_round([(i,0) for i in (0,1)],opts.gpus,
            lambda item,gpu:execute(f'extract_{item[0]}','extract',gpu,ids=all_ids[item[0]::2]),cancel,deadline,
            lambda item,exc:(_ for _ in ()).throw(GlobalFailure(f'Shared extraction failed: {exc}')))
        raw={g:refs[g]['raw'] for g in 'AB'}
        for group in 'CDEF':
            ensure_resources(deadline)
            def merge_action(attempt,group=group):
                path=raw_root/group/attempt.name
                merge(cache,context['train_ids'],path,base['raw'],group,signatures[group],cache_identity); return {'raw':str(path)},[path]
            raw[group]=stages.run('merge_'+group,merge_action)['raw']
        replay={}
        tasks=[('SMFA',s) for s in SEEDS]+[('A',42),('B',42)]
        def replay_task(item,gpu):
            group,seed=item; historic=smfa[seed] if group=='SMFA' else refs[group]
            current=dict(historic); current.update(execute(f'replay_{group}_{seed}','eval',gpu,group,seed,historic))
            mpsa.require_baseline_replay(ab.read(historic['metrics']),ab.read(current['metrics']))
            replay[item]=current
        run_round(tasks,opts.gpus,replay_task,cancel,deadline,lambda item,exc:(_ for _ in ()).throw(GlobalFailure(f'Historical replay failed: {exc}')))
        for group in 'AB': commit_report(f'{group}_42',compare(replay['SMFA',42],replay[group,42],context['val_ids']))
        flush()
        def task(item,gpu):
            group,seed=item; key=f'{group}_{seed}'; current={'raw':raw[group]}
            if key in reports: return
            failure_file=output/'task_failures'/f'{key}.json'
            prior=ab.read(failure_file) if failure_file.exists() else dict(attempts=0)
            if prior['attempts']>=2 or prior.get('oom'): raise TaskFailure(f'Prior retry budget exhausted: {key}')
            for retry in range(prior['attempts'],2):
                try:
                    current.update(execute(f'train_{key}','train',gpu,group,seed,current))
                    current.update(execute(f'eval_{key}','eval',gpu,group,seed,current,accepted=True))
                    value=compare(replay['SMFA',seed],current,context['val_ids'])
                    value['per_image_files']=dict(smfa=replay['SMFA',seed]['per_image'],candidate=current['per_image'])
                    commit_report(key,value)
                    flush(); return
                except TaskFailure as exc:
                    ab.write(failure_file,dict(attempts=retry+1,oom=isinstance(exc,TaskOOM),error=str(exc)))
                    if isinstance(exc,TaskOOM) or retry==1: raise
                    ensure_resources(deadline)
        for seed in SEEDS: run_round(task_order(seed),opts.gpus,task,cancel,deadline,fail)
        stages.verify_all(); gmp.verify(records,'Historical artifacts'); gmp.verify(context['inputs'],'Inputs')
        for stem in context['train_ids']+context['val_ids']:
            if cache_read(cache/('train' if stem in train_set else 'val')/stem,cache_identity[stem]) is None:
                raise GlobalFailure('Shared cache disappeared')
        if ab.source_identity()!=context['code']: raise GlobalFailure('Source changed')
        status='complete' if len(reports)==18 else 'complete_with_failures'
    except Deferred as exc: status='deferred'; error=str(exc)
    except BaseException as exc:
        cancel.set(); status='incomplete'; error=str(exc); raise
    finally:
        flush(); ab.write(output/'status.json',dict(status=status,error=error,completed=len(reports),expected=18))
        signal.signal(signal.SIGTERM,previous_signal); lock.unlink()
    print(f'[{status}] {output}/summary.json',flush=True)


if __name__=='__main__': main()
