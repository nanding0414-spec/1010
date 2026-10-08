#!/usr/bin/env python3
"""Disposable SMFA/PMFA experiment, sequential or dual GPU. No production source edits.

Run --help or see docs/PROJECT_STORAGE_MANUAL.md, experimental PMFA section.
Heavy dependencies are imported in workers, after CUDA_VISIBLE_DEVICES is set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import shutil
import subprocess
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
POLICY = 'standard_vit_mlp_fp32_v1'
EXPERIMENT_FORMAT = 3001


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identities(paths):
    return [dict(path=str(p.resolve()), size=p.stat().st_size, sha256=digest(p))
            for p in sorted(set(map(Path, paths)))]


def files_under(paths):
    return [f for p in map(Path, paths) for f in (sorted(p.rglob('*')) if p.is_dir() else [p]) if f.is_file()]


def source_identity():
    files = list(ROOT.glob('*.py'))
    for folder in ('src', 'configs', 'project_utils', 'tools', 'sam3-main/sam3'):
        files.extend((ROOT / folder).rglob('*.py'))
    files.extend((ROOT / 'configs').glob('*.txt'))
    files.append(ROOT / 'tools/proposal_demo/voc_classes.json')
    return identities(files)


@dataclass(frozen=True)
class PMFA:
    core_threshold: float = .9
    steps: int = 2
    rel_margin: float = .12
    temperature: float = .2
    alpha: float = .5
    min_support: int = 3

    def __post_init__(self):
        for name in ('core_threshold', 'rel_margin', 'temperature', 'alpha'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.core_threshold <= 1 or not 0 <= self.alpha <= 1:
            raise ValueError('core_threshold and alpha must be in [0,1]')
        if self.temperature <= 0 or self.rel_margin < 0:
            raise ValueError('temperature must be positive; rel_margin nonnegative')
        if type(self.steps) is not int or self.steps < 0 or type(self.min_support) is not int or self.min_support < 1:
            raise ValueError('steps/min_support must be nonnegative/positive integers')


def signature(base, mode, params):
    if mode not in ('smfa', 'pmfa'):
        raise ValueError(mode)
    return dict(base, format_version=EXPERIMENT_FORMAT, experimental_pooling={
        'version': 2, 'mode': mode, 'parameters': asdict(params) if mode == 'pmfa' else {},
        'coverage_policy': 'fp32_tolerance_1e-6_pmfa_clamp_preserve_smfa',
        'precision_policy': POLICY, 'mask_chunk_size': 16,
    })


def pool(tokens, weights, params, original, stats=None):
    """One region. original(tokens, weights) is the exact existing SMFA path."""
    import torch
    import torch.nn.functional as F
    if tokens.ndim != 2 or weights.ndim != 1 or len(tokens) != len(weights):
        raise ValueError('Expected tokens[N,D] and mask coverage[N]')
    if not torch.isfinite(tokens).all() or not torch.isfinite(weights).all():
        raise ValueError('Non-finite token or coverage')
    # OpenCV INTER_AREA can round an all-one region slightly above one in FP32.
    # Reject real invalid values, but permit rounding at the mathematical bounds.
    if ((weights < -1e-6) | (weights > 1+1e-6)).any():
        raise ValueError(f'Coverage outside [0,1] tolerance=1e-6: '
                         f'min={float(weights.min()):.9g}, max={float(weights.max()):.9g}')
    stats = stats if stats is not None else Counter()
    stats['regions'] += 1
    if params.alpha == 0:
        stats['fallback_alpha_zero'] += 1
        return original(tokens, weights)  # Preserve exact original SMFA arithmetic.
    original_weights = weights
    outside = (weights < 0) | (weights > 1)
    if outside.any():
        stats['coverage_roundoff_regions'] += 1
        stats['coverage_roundoff_tokens'] += int(outside.sum())
        weights = weights.clamp(0, 1)
    valid = weights > 0
    n = int(valid.sum())
    stats['valid_support_sum'] += n
    stats['core_support_sum'] += int((weights >= params.core_threshold).logical_and(valid).sum())
    if n < params.min_support or float(weights.sum()) < 1e-6:
        stats['fallback_small_or_empty'] += 1
        return original(tokens, original_weights)
    feats, m = tokens[valid].float(), weights[valid].float()
    unit = F.normalize(feats, dim=-1, eps=1e-6)
    core = m >= params.core_threshold
    if not core.any():
        stats['core_top3_initializations'] += 1
        core = torch.zeros_like(m, dtype=torch.bool)
        core[torch.topk(m, min(3, n)).indices] = True
    q = (unit[core] * m[core, None]).sum(0)
    if q.norm() < 1e-6:
        stats['fallback_zero_proxy'] += 1
        return original(tokens, original_weights)
    q = F.normalize(q, dim=0, eps=1e-6)
    for _ in range(params.steps):
        sim = unit @ q
        selected = sim >= sim.max() - params.rel_margin
        if int(selected.sum()) < min(params.min_support, n):
            selected = torch.zeros_like(m, dtype=torch.bool)
            selected[torch.topk(sim, min(params.min_support, n)).indices] = True
        candidate = (unit[selected] * m[selected, None]).sum(0)
        if candidate.norm() < 1e-6:
            stats['zero_update_breaks'] += 1
            break
        q = F.normalize(candidate, dim=0, eps=1e-6)
    attention = torch.softmax(unit @ q / params.temperature + m.clamp_min(1e-6).log(), dim=0)
    stats['attention_effective_support_sum'] += float(1 / attention.square().sum())
    stats['proxy_regions'] += 1
    mean = (feats * m[:, None]).sum(0) / (m.sum() + 1e-8)
    result = F.normalize((1-params.alpha)*mean + params.alpha*(feats*attention[:, None]).sum(0), dim=0)
    if not torch.isfinite(result).all():
        raise ValueError('PMFA produced non-finite output')
    return result


def install_precision():
    """Must run before importing feature/proposal modules that bind the builder."""
    import torch
    import sam3.model_builder as builder
    from src.sam3_precision import use_standard_vit_mlp
    actual = Path(builder.__file__).resolve()
    if not actual.is_relative_to((ROOT / 'sam3-main/sam3').resolve()):
        raise RuntimeError(f'Wrong SAM3 import: {actual}')
    original = builder.build_sam3_image_model
    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        count = use_standard_vit_mlp(model)
        if not count:
            raise RuntimeError('No supported SAM3 ViT MLP found; refusing unknown precision policy')
        model.float()
        print(f'[FP32] {count} ViT MLP instances patched; {actual}', flush=True)
        return model
    builder.build_sam3_image_model = build
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def install_pooling(mode, params, stats):
    import numpy as np
    import torch
    from src.feature_extractor import BaseFeatureExtractor
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    import src.feature_signature as fs
    fs.REQUIRED_SIGNATURE_KEYS.add('experimental_pooling')
    old_signature = SAM3FeatureExtractor.feature_signature
    SAM3FeatureExtractor.feature_signature = lambda self: signature(old_signature(self), mode, params)
    if mode == 'smfa':
        return
    old_pool = BaseFeatureExtractor._pool_tokens
    def one(self, tokens, mask):
        if self.context_subtract != 0:
            raise ValueError('This experiment requires context_subtract=0')
        weights = torch.as_tensor(mask, dtype=torch.float32, device=tokens.device)
        return pool(tokens, weights, params,
                    lambda t, w: old_pool(self, t, w.detach().cpu().numpy()), stats)
    def batch(self, rgb, masks):
        if not masks:
            return []
        inp, transform = self._preprocess_image(rgb)
        tokens, (hp, wp) = self._forward_tokens(inp)
        if tokens is None or tokens.shape[0] != 1:
            raise RuntimeError('PMFA requires one-image spatial tokens')
        tok = tokens[0]
        output = []
        # No [regions,tokens,channels] expansion; at most 16 mask vectors alive.
        for start in range(0, len(masks), 16):
            coverage = [self._downsample_mask_to_patches(np.asarray(m, dtype=bool), hp, wp, transform)
                        for m in masks[start:start+16]]
            output.extend(one(self, tok, w).detach().cpu().numpy() for w in coverage)
        return output
    BaseFeatureExtractor._pool_tokens = one
    BaseFeatureExtractor.extract_features_batch = batch
    print('[Experiment] PMFA pooling installed (base constructor may still log pooling=smfa); '
          'coverage tolerance=1e-6, signature version=2',flush=True)


def get_ids(path):
    ids = Path(path).read_text(encoding='utf-8-sig').split()
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[\w-]+', i) for i in ids):
        raise ValueError(f'Empty, duplicate or unsafe IDs: {path}')
    return ids


def image_index(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.stem in result:
                raise ValueError(f'Duplicate image ID: {path.stem}')
            result[path.stem] = str(path.resolve())
    return result


def validate_cache(directory, kind, ids, sizes):
    import numpy as np
    from PIL import Image
    errors = []
    if not directory:
        return ['not supplied']
    for stem in ids:
        path = Path(directory) / (stem + ('.png' if kind == 'pseudo' else '.npz'))
        try:
            w, h = sizes[stem]
            if kind == 'pseudo':
                with Image.open(path) as im:
                    array = np.asarray(im)
                    if array.shape != (h,w) or not np.isin(array, list(range(21))+[255]).all():
                        raise ValueError('invalid VOC PNG dimensions/IDs')
            else:
                with np.load(path, allow_pickle=False) as data:
                    m, s = data['masks'], data['scores']
                    if m.ndim != 3 or s.shape != (len(m),) or not np.isfinite(s).all():
                        raise ValueError('invalid NPZ shape/scores')
                    if m.shape[1:] != (h,w) and not (len(m)==0 and m.shape[1:]==(1,1)):
                        raise ValueError('proposal/image dimensions differ')
                    if not np.all((m==0)|(m==1)):
                        raise ValueError('nonbinary proposal')
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            errors.append(f'{path}: {exc}')
    return errors


def check_gpu(gpu):
    # nvidia-smi utilization includes a recent sampling window after a worker exits.
    # Only low-memory utilization spikes receive a bounded settling period.
    for attempt in range(16):
        value = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        name, total, used, utilization = [x.strip() for x in value.split(',')]
        total,used,utilization=map(float,(total,used,utilization))
        if not all(math.isfinite(x) and x>=0 for x in (total,used,utilization)):
            raise RuntimeError(f'Invalid GPU telemetry: {value}')
        if used>1000 or (utilization>10 and attempt==15):
            raise RuntimeError(f'Physical GPU {gpu} occupied: {value}')
        if utilization<=10:
            return dict(physical_gpu=gpu,name=name,total_mib=total,used_mib=used,utilization=utilization)
        print(f'[GPU] Physical GPU {gpu}: {used:g} MiB, utilization {utilization:g}%; '
              f'waiting for utilization to settle ({attempt*2}/30s)',flush=True)
        time.sleep(2)


def preflight(args):
    import numpy as np
    from PIL import Image
    required = [args.train_list, args.val_list, args.imagelevel_json, args.checkpoint,
                ROOT/'sam3-main/sam3/model_builder.py']
    missing = [str(p) for p in required if not p.is_file()]
    if not args.image_dir.is_dir():
        missing.append(str(args.image_dir))
    if missing:
        raise FileNotFoundError('Missing inputs (no automatic download):\n'+'\n'.join(missing))
    train, val = get_ids(args.train_list), get_ids(args.val_list)
    if set(train)&set(val):
        raise ValueError('Training/final validation overlap')
    rows = read(args.imagelevel_json)['images']
    json_ids = [Path(r['file_name']).stem for r in rows]
    if len(json_ids)!=len(set(json_ids)) or set(json_ids)!=set(train):
        raise ValueError('Image-level JSON must exactly cover the training split')
    for r in rows:
        labels = r.get('labels')
        if not isinstance(labels,list) or any(type(i) is not int or not 0<=i<=20 for i in labels):
            raise ValueError(f'Invalid VOC image-level labels: {r["file_name"]}')
    images = image_index(args.image_dir)
    missing = [i for i in train+val if i not in images]
    if missing:
        raise FileNotFoundError(f'Missing images: {missing[:20]} (total {len(missing)})')
    if args.gt_dir is None:
        args.gt_dir = next((args.dataset_root/name for name in ('SegmentationClassAug','SegmentationClass')
                            if all((args.dataset_root/name/f'{i}.png').is_file() for i in val)), None)
    if args.gt_dir is None:
        raise FileNotFoundError('No complete val GT directory; pass --gt-dir')
    missing = [str(args.gt_dir/f'{i}.png') for i in val if not (args.gt_dir/f'{i}.png').is_file()]
    if missing:
        raise FileNotFoundError('Missing val GT:\n'+'\n'.join(missing[:20]))
    sizes = {}; val_set = set(val)
    for i in train+val:
        with Image.open(images[i]) as im:
            im.load(); sizes[i] = list(im.size)
        if i in val_set:
            with Image.open(args.gt_dir/f'{i}.png') as im:
                a = np.asarray(im)
                if list(im.size)!=sizes[i] or a.ndim!=2 or not np.isin(a,list(range(21))+[255]).all():
                    raise ValueError(f'Invalid GT: {i}')
    selected = {}
    for kind, directory, ids in [('pseudo',args.pseudo_dir,train),('proposals',args.proposals_dir,val)]:
        errors = validate_cache(directory, kind, ids, sizes)
        selected[kind] = str(directory) if not errors else None
        print(f'[Input] {kind}: '+(f'reuse {directory}' if not errors else f'generate fresh full cache; {len(errors)} issue(s), first: {errors[0]}'), flush=True)
    missing_modules = [m for m in ('torch','torchvision','cv2','faiss','torchmetrics','timm','einops','iopath')
                       if importlib.util.find_spec(m) is None]
    if missing_modules:
        raise RuntimeError(f'Activate moduseg_sam3; missing dependencies: {missing_modules}')
    gpu = {str(i):check_gpu(i) for i in sorted(set([args.gpu]+(args.parallel_gpus or [])))}
    free = shutil.disk_usage(ROOT).free
    print(f'[Preflight] project filesystem free={free/2**30:.1f} GiB (not a capacity estimate)',flush=True)
    if free < 2**30:
        raise OSError('Less than 1 GiB free; make space before running')
    inputs = required + [Path(images[i]) for i in train+val] + [args.gt_dir/f'{i}.png' for i in val]
    for kind, ids in [('pseudo',train),('proposals',val)]:
        if selected[kind]:
            inputs += [Path(selected[kind])/(i+('.png' if kind=='pseudo' else '.npz')) for i in ids]
    print('[Preflight] hashing input files and code; this can take time', flush=True)
    return dict(train_ids=train,val_ids=val,images={i:images[i] for i in train+val},sizes=sizes,
                selected=selected,inputs=identities(inputs),code=source_identity(),gpu=gpu)


def base_env(gpu):
    env = {k:v for k,v in os.environ.items() if not k.startswith('OVERRIDE_')}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        OVERRIDE_DATASET_TYPE='voc',OVERRIDE_FEATURE_BACKBONE='sam3',OVERRIDE_VOC_TRAIN_SPLIT='trainaug',
        OVERRIDE_SAM3_RESOLUTION='1008',OVERRIDE_SAM3_FEATURE_SOURCE='trunk_fpn',OVERRIDE_SAM3_FEATURE_LEVEL='-1',
        OVERRIDE_SAM3_TOKEN_NORM='layernorm',OVERRIDE_SAM3_CONTEXT_SUBTRACT='0',OVERRIDE_SAM3_USE_BF16='false',
        OVERRIDE_FEATURE_ADAPTER_CKPT='',OVERRIDE_SEED='42',OVERRIDE_MIN_MASK_AREA='300',
        OVERRIDE_BUILD_SPLIT_CONNECTED_COMPONENTS='true',OVERRIDE_FILTER_FOREGROUND_DROP_RATIO='0.1',
        OVERRIDE_FILTER_BACKGROUND_DROP_RATIO='0',OVERRIDE_INF_LOAD_PROPOSALS='true',
        OVERRIDE_INF_RETRIEVAL_MODE='balanced',OVERRIDE_INF_TOPK_NEIGH='25',OVERRIDE_INF_CLASS_TOPM='5',
        OVERRIDE_INF_RETRIEVAL_SEARCH_MULTIPLIER='2',OVERRIDE_INF_CENTROID_WEIGHT='0.4',
        OVERRIDE_INF_RETRIEVAL_MARGIN_THR='0',OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT='0.2',
        OVERRIDE_INF_NMS_IOU='0.4',OVERRIDE_INF_INSTANCE_SCORE_THR='0.02',OVERRIDE_INF_VIS_LIMIT='0',
        OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK='false',OVERRIDE_INF_FAISS_USE_GPU='true',
        OVERRIDE_INF_FAISS_DEVICE='0',OVERRIDE_INF_FAISS_EXACT_SEARCH_THRESHOLD='100000')
    return env


def launch(job, attempt, env, log, cancel=None):
    jobfile = attempt/'job.json';write(jobfile,job)
    command = [sys.executable,'-u',str(SCRIPT),'--worker',str(jobfile)]
    write(attempt/'command.json',dict(command=command,environment={k:v for k,v in env.items()
          if k.startswith('OVERRIDE_') or k=='CUDA_VISIBLE_DEVICES'}))
    log.parent.mkdir(parents=True,exist_ok=True)
    check_gpu(int(env['CUDA_VISIBLE_DEVICES']))
    process = None
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if main_thread else None
    def interrupt(*_): raise KeyboardInterrupt('Launcher interrupted')
    if main_thread: signal.signal(signal.SIGTERM,interrupt)
    try:
        if cancel is not None and cancel.is_set(): raise RuntimeError('Parallel run cancelled')
        with log.open('x',encoding='utf-8') as stream:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                       start_new_session=(os.name=='posix'))
            print(f'[Worker] pid={process.pid}; log={log}',flush=True)
            while process.poll() is None:
                if cancel is not None and cancel.is_set(): raise RuntimeError('Parallel run cancelled')
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f'{job["stage"]} failed ({process.returncode}); inspect {log}')
    finally:
        if process and process.poll() is None:
            if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
            else: process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
                else: process.kill()
                process.wait()
        if main_thread: signal.signal(signal.SIGTERM,previous)


class Stages:
    def __init__(self, output, resume):
        self.output,self.resume = Path(output),resume

    def verify_all(self):
        for marker in self.output.glob('*/complete.json'):
            value=read(marker)
            if identities([r['path'] for r in value['artifacts']])!=value['artifacts']:
                raise ValueError(f'Completed artifacts changed during run: {marker.parent.name}')

    def run(self,key,action):
        folder=self.output/key;marker=folder/'complete.json'
        if marker.exists():
            if not self.resume: raise FileExistsError(marker)
            value=read(marker)
            if identities([r['path'] for r in value['artifacts']])!=value['artifacts']:
                raise ValueError(f'Completed artifacts changed: {key}')
            print(f'[Resume] {key}',flush=True)
            return value['result']
        folder.mkdir(parents=True,exist_ok=True)
        number=1
        while (folder/f'attempt_{number:03d}').exists(): number+=1
        attempt=folder/f'attempt_{number:03d}';attempt.mkdir()
        started=time.monotonic()
        try:
            result,artifacts=action(attempt)
            write(marker,dict(result=result,artifacts=identities(files_under(artifacts)),seconds=time.monotonic()-started))
            return result
        except BaseException as exc:
            write(attempt/'failed.json',dict(error=str(exc),type=type(exc).__name__,seconds=time.monotonic()-started))
            raise


def worker(job):
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'sam3-main'))
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    install_precision()
    stats=Counter();started=time.monotonic();status='failed'
    try:
        mode=job.get('mode','smfa');params=PMFA(**job['pmfa'])
        if job['stage'] not in ('pseudo','proposals'):
            install_pooling(mode,params,stats)
        import configs.config as cfg
        cfg.DATASET_ROOT=cfg.INF_DATASET_ROOT=job['dataset_root']
        from project_utils.seed_utils import set_global_seed
        set_global_seed(42)
        stage=job['stage']
        if stage=='probe':
            import numpy as np
            from PIL import Image
            from src.feature_extractor_factory import build_feature_extractor
            model=build_feature_extractor(torch.device('cuda:0'),use_interactivity=False)
            with Image.open(job['images'][job['train_ids'][0]]) as im: rgb=np.array(im.convert('RGB'))
            value=model.extract_features_batch(rgb,[np.ones(rgb.shape[:2],dtype=bool)])[0]
            if not np.isfinite(value).all() or np.linalg.norm(value)<1e-6:
                raise ValueError('FP32 forward probe failed')
            write(Path(job['attempt'])/'probe.json',dict(signature=model.feature_signature(),dimension=len(value)))
        elif stage=='pseudo':
            from configs.experiment import PSEUDO,cli_options
            recipe=dict(PSEUDO,dataset='voc',split='trainaug',no_bf16=True)
            from tools.generate_pseudo_masks_sam3 import main
            code=main([*cli_options(recipe),'--dataset-root',job['dataset_root'],'--image-dir',job['image_dir'],
                  '--imagelevel-json',job['imagelevel_json'],'--ckpt',job['checkpoint'],
                  '--output-dir',job['destination'],'--experiment-dir',str(Path(job['attempt'])/'generation'),
                  '--no-evaluate','--no-resume'])
            if code: raise RuntimeError('Pseudo generator reported failure')
        elif stage=='proposals':
            from run_generate_proposals_active import main
            code=main(['--backend','active','--precision','fp32','--device','cuda:0','--seed','42',
                  '--image-dir',job['image_dir'],'--split',job['val_list'],'--checkpoint',job['checkpoint'],
                  '--output-dir',job['destination']])
            if code: raise RuntimeError('Proposal exporter reported failure')
        elif stage=='build':
            from src.region_extractors import get_regions_from_pseudo
            from src.feature_extractor_factory import build_feature_extractor
            from src.build_feature_bank import create_dirs,save_class_mapping,FeatureBank
            from src.feature_signature import save_feature_signature
            from src.filter_stage import run_filter_stage
            from project_utils.config_logger import record_config
            if Path(cfg.OUTPUT_ROOT).exists(): raise FileExistsError(cfg.OUTPUT_ROOT)
            create_dirs();save_class_mapping(21,cfg.CLS_PATH);record_config(cfg.META_DIR,'experimental_pmfa_ab.py')
            model=build_feature_extractor(torch.device('cuda:0'),use_interactivity=False)
            save_feature_signature(cfg.META_DIR,model.feature_signature())
            bank=FeatureBank();counts={}
            for pos,stem in enumerate(job['train_ids'],1):
                regions=get_regions_from_pseudo(job['images'][stem],300,job['pseudo'])
                before=model._feature_failure_count
                vectors=model.extract_features_for_regions(job['images'][stem],regions,0) if regions else {}
                if before!=model._feature_failure_count or sum(map(len,vectors.values()))!=len(regions):
                    raise RuntimeError(f'Incomplete extraction: {stem}')
                for cid,values in vectors.items():
                    for value in values:
                        if not bank._is_valid_feature(value): raise ValueError(f'Invalid feature: {stem}')
                        bank.add_feature('background' if cid==0 else cfg.DATASET_CLASSES[cid-1],value,stem)
                counts[stem]=len(regions)
                print(f'[Build {mode}] {pos}/{len(job["train_ids"])} {stem} {len(regions)}',flush=True)
            if not sum(counts.values()): raise ValueError('Empty raw bank')
            bank.save_stats();bank.build_indices();run_filter_stage()
            write(Path(cfg.META_DIR)/'coverage.json',dict(ids=job['train_ids'],region_counts=counts))
        elif stage=='train':
            # The existing trainer validates source signatures and exports the best checkpoint.
            raw_signature=read(Path(job['raw'])/'meta/feature_signature.json')
            if (raw_signature.get('format_version')!=EXPERIMENT_FORMAT or
                raw_signature.get('experimental_pooling')!=signature({},mode,params)['experimental_pooling']):
                raise ValueError('Training bank does not match this experimental method/parameters/precision')
            args=['--source-bank',job['raw'],'--train-list',job['train_list'],
                  '--output-dir',job['weights'],'--export-bank',job['bank'],'--device','cuda:0',
                  '--recipe','v1','--classifier','linear','--selection-metric','knn','--eval-topk','5',
                  '--feature-dropout','0','--consistency-weight','0','--label-smoothing','0',
                  '--num-layers','3','--hidden-dim','512','--output-dim','256','--epochs','50',
                  '--steps-per-epoch','100','--classes-per-batch','8','--instances-per-class','4',
                  '--lr','0.001','--temperature','0.1','--contrastive-weight','0.5','--val-fraction','0.2',
                  '--seed','42','--cosine-scale','16','--eval-class-topm','5','--eval-centroid-weight','0.4',
                  '--eval-search-multiplier','2']
            if job.get('split_file'): args+=['--split-file',job['split_file']]
            sys.argv=[str(ROOT/'tools/train_feature_adapter.py'),*args]
            runpy.run_path(sys.argv[0],run_name='__main__')
        elif stage=='eval':
            import src.inference as inf
            inf.load_eval_samples=lambda _: [dict(id=i,img=job['images'][i],gt=str(Path(job['gt_dir'])/f'{i}.png')) for i in job['val_ids']]
            import run_inference
            run_inference.main(['--strict-eval','--eval-limit','0','--cleanup-config',job['cleanup'],
                                '--results-json',job['metrics'],'--output-dir',str(Path(job['attempt'])/'predictions')])
        else: raise ValueError(stage)
        torch.cuda.synchronize();status='complete'
    finally:
        report=dict(status=status,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),pooling_stats=dict(stats),
                    torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),precision_policy=POLICY)
        write(Path(job['attempt'])/'worker_report.json',report)


def bank_members(root):
    from src.feature_bank_storage import load_feature_matrix_npz
    members={}
    for file in sorted((Path(root)/'features').glob('*/*.npz')):
        _,ids=load_feature_matrix_npz(file)
        members[file.parent.name]=set(map(str,ids))
    return members


def compare(results, context):
    a,b=read(results['smfa']['metrics']),read(results['pmfa']['metrics'])
    for value in (a,b):
        if value['evaluated_ids']!=context['val_ids'] or value['evaluated_count']!=len(context['val_ids']) or value['eval_limit']!=0:
            raise ValueError('Incomplete/mismatched final evaluation coverage')
        if not math.isfinite(value['mIoU']) or not 0<=value['mIoU']<=1: raise ValueError('Invalid mIoU')
    for key in ('cleanup','proposals_dir','proposal_count_before','proposal_count_after'):
        if a[key]!=b[key]: raise ValueError(f'Unfair comparison: {key}')
    ignore={'INF_FEATURE_BANK_ROOT','INF_SAVE_DIR'}
    if {k:v for k,v in a['inference_settings'].items() if k not in ignore}!={k:v for k,v in b['inference_settings'].items() if k not in ignore}:
        raise ValueError('Inference settings differ')
    for mode,value in [('smfa',a),('pmfa',b)]:
        sig=value['feature_signature']
        if sig['experimental_pooling']['mode']!=mode or sig!=read(Path(results[mode]['bank'])/'meta/feature_signature.json'):
            raise ValueError('Wrong evaluated feature signature')
    members=[bank_members(results[m]['raw']) for m in ('smfa','pmfa')]
    membership={c:dict(smfa=len(members[0].get(c,set())),pmfa=len(members[1].get(c,set())),
                       common=len(members[0].get(c,set())&members[1].get(c,set())),
                       smfa_only=sorted(members[0].get(c,set())-members[1].get(c,set())),
                       pmfa_only=sorted(members[1].get(c,set())-members[0].get(c,set())))
                for c in sorted(set(members[0])|set(members[1]))}
    delta=100*(b['mIoU']-a['mIoU'])
    per_class={c:dict(smfa=v,pmfa=b['per_class_iou'][c],delta_pp=100*(b['per_class_iou'][c]-v)) for c,v in a['per_class_iou'].items()}
    splits={m:read(Path(results[m]['weights'])/'training_report.json') for m in ('smfa','pmfa')}
    return dict(status='complete',smfa_miou=a['mIoU'],pmfa_miou=b['mIoU'],delta_pp=delta,
                outcome='提高' if delta>1e-8 else '下降' if delta< -1e-8 else '持平',
                note='Single seed; not statistical significance. Each method filters its own bank.',
                per_class_iou=per_class,background=per_class.get('background'),bank_membership=membership,
                evaluated_ids=context['val_ids'],confusion_matrices=dict(smfa=a['confusion_matrix'],pmfa=b['confusion_matrix']),
                internal_split_coverage={m:{k:splits[m][k] for k in ('train_images','validation_images')} for m in splits})


def parallel_arms(execute, shared, cleanup, results, on_complete):
    """One worker per GPU; PMFA training depends on committed SMFA training."""
    cancel=threading.Event();ready=threading.Event();baseline={}
    def arm(mode):
        raw=execute('build',mode,_cancel=cancel,pseudo=shared['pseudo'])
        split=None
        if mode=='pmfa':
            while not ready.wait(0.2):
                if cancel.is_set(): raise RuntimeError('Cancelled while waiting for SMFA split')
            split=str(Path(baseline['weights'])/'training_report.json')
        trained=execute('train',mode,_cancel=cancel,raw=raw['raw'],split_file=split)
        if mode=='smfa':
            baseline.update(trained);ready.set()
        evaluated=execute('eval',mode,_cancel=cancel,bank=trained['bank'],weights=trained['weights'],
                          proposals=shared['proposals'],cleanup=str(cleanup))
        return dict(raw=raw['raw'],weights=trained['weights'],bank=trained['bank'],metrics=evaluated['metrics'])
    previous=signal.getsignal(signal.SIGTERM)
    def interrupt(*_):
        cancel.set()
        raise KeyboardInterrupt('Parallel launcher interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    futures={}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                futures={executor.submit(arm,mode):mode for mode in ('smfa','pmfa')}
                pending=set(futures)
                while pending:
                    done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        results[futures[future]]=future.result()
                        on_complete()
            finally:
                cancel.set()  # Workers exit their own process groups before shutdown joins.
    finally:
        # Preserve a completed arm even if its peer failed in the same polling cycle.
        for future,mode in futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                results[mode]=future.result()
        signal.signal(signal.SIGTERM,previous)


def run_pipeline(args,context):
    output=ROOT/'experiments'/args.version
    roots={kind:ROOT/'storage'/kind/'voc'/args.version for kind in ('pseudo_labels','proposals','raw_banks','mlp_banks')}
    roots.update(weights=ROOT/'weights/sam3_adapter'/args.version,logs=ROOT/'logs'/args.version)
    params=PMFA(**{k:getattr(args,k) for k in asdict(PMFA())})
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','preflight_only','worker')}
    manifest=dict(version=1,options=options,inputs=context['inputs'],code=context['code'],
                  selected=context['selected'],pmfa=asdict(params),precision_policy=POLICY,
                  environment=base_env(args.gpu))
    # Never persist unrelated environment variables/secrets.
    manifest['environment']={k:v for k,v in manifest['environment'].items() if k.startswith('OVERRIDE_') or k=='CUDA_VISIBLE_DEVICES'}
    if output.exists():
        if not args.resume or read(output/'manifest.json')!=manifest:
            raise ValueError('Existing version differs, or --resume missing; choose a fresh --version')
    else:
        collisions=[str(p) for p in roots.values() if p.exists()]
        if collisions: raise FileExistsError(f'Output version already in use: {collisions}')
        write(output/'manifest.json',manifest)
    lock=output/'launcher.lock'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError(f'Launcher lock exists: {lock}; verify its PID is dead before manually removing stale lock')
    with os.fdopen(fd,'w') as stream: stream.write(str(os.getpid()))
    stage=Stages(output,args.resume);results={}
    try:
        write(output/'artifacts.json',dict(owned_roots=[str(output),*[str(p) for p in roots.values()]],
                                         read_only_inputs=context['selected'],automatic_deletion=False))
        common=dict(pmfa=asdict(params),dataset_root=str(args.dataset_root),image_dir=str(args.image_dir),
                    imagelevel_json=str(args.imagelevel_json),checkpoint=str(args.checkpoint),train_list=str(args.train_list),
                    val_list=str(args.val_list),gt_dir=str(args.gt_dir),train_ids=context['train_ids'],
                    val_ids=context['val_ids'],images=context['images'])
        env=base_env(args.gpu);env['OVERRIDE_SAM3_CKPT']=str(args.checkpoint)
        gpu_map=dict(zip(('smfa','pmfa'),args.parallel_gpus or [args.gpu,args.gpu]))
        def execute(kind,mode='smfa',_cancel=None,**extras):
            if _cancel is not None and _cancel.is_set(): raise RuntimeError('Parallel run cancelled')
            key=f'{mode}_{kind}' if kind in ('build','train','eval') or (kind=='probe' and args.parallel_gpus) else kind
            def action(attempt):
                job=dict(common,stage=kind,mode=mode,attempt=str(attempt),**extras)
                current=dict(env);artifacts=[attempt];result={}
                if kind in ('probe','build','train','eval'):
                    current['CUDA_VISIBLE_DEVICES']=str(gpu_map[mode])
                if kind in ('pseudo','proposals'):
                    dest=roots['pseudo_labels' if kind=='pseudo' else 'proposals']/attempt.name
                    if dest.exists(): raise FileExistsError(f'Refusing to overwrite generated input: {dest}')
                    job['destination']=str(dest);result={kind:str(dest)};artifacts.append(dest)
                elif kind=='build':
                    dest=roots['raw_banks']/mode/attempt.name
                    current.update(OVERRIDE_OUTPUT_ROOT=str(dest),OVERRIDE_PSEUDO_MASK_ROOT=extras['pseudo'],OVERRIDE_BUILD_IMAGE_DIR=str(args.image_dir))
                    result={'raw':str(dest)};artifacts.append(dest)
                elif kind=='train':
                    w=roots['weights']/mode/attempt.name;b=roots['mlp_banks']/mode/attempt.name
                    job.update(weights=str(w),bank=str(b));result=dict(weights=str(w),bank=str(b));artifacts.extend([w,b])
                elif kind=='eval':
                    metrics=attempt/'metrics.json';job['metrics']=str(metrics);result={'metrics':str(metrics)}
                    current.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(extras['weights'])/'adapter.pt'),
                                   OVERRIDE_INF_FEATURE_BANK_ROOT=extras['bank'],OVERRIDE_INF_PROPOSALS_DIR=extras['proposals'])
                log=roots['logs']/f'{key}_{attempt.name}.log'
                if _cancel is None: launch(job,attempt,current,log)
                else: launch(job,attempt,current,log,cancel=_cancel)
                artifacts.append(log)
                report=read(attempt/'worker_report.json')
                if report['status']!='complete': raise RuntimeError('Worker did not complete')
                if kind in ('pseudo','proposals'):
                    issues=validate_cache(job['destination'],kind,context['train_ids' if kind=='pseudo' else 'val_ids'],context['sizes'])
                    if issues: raise ValueError(f'Generated cache incomplete: {issues[:10]}')
                result['worker_report']=str(attempt/'worker_report.json')
                return result,artifacts
            return stage.run(key,action)
        execute('probe')
        if args.parallel_gpus: execute('probe','pmfa')
        shared={}
        for kind in ('pseudo','proposals'):
            shared[kind]=context['selected'][kind] or execute(kind)[kind]
        from configs.experiment import CLEANUP
        cleanup=output/'cleanup.json'
        if cleanup.exists() and read(cleanup)!=CLEANUP: raise ValueError('Cleanup recipe changed')
        if not cleanup.exists(): write(cleanup,CLEANUP)
        if args.parallel_gpus:
            parallel_arms(execute,shared,cleanup,results,
                          lambda: write(output/'status.json',dict(status='in_progress',completed=results)))
        for mode in (() if args.parallel_gpus else ('smfa','pmfa')):
            raw=execute('build',mode,pseudo=shared['pseudo'])
            split=str(Path(results['smfa']['weights'])/'training_report.json') if mode=='pmfa' else None
            trained=execute('train',mode,raw=raw['raw'],split_file=split)
            evaluated=execute('eval',mode,bank=trained['bank'],weights=trained['weights'],proposals=shared['proposals'],cleanup=str(cleanup))
            results[mode]=dict(raw=raw['raw'],weights=trained['weights'],bank=trained['bank'],metrics=evaluated['metrics'])
            write(output/'status.json',dict(status='in_progress',completed=results))
        if identities([r['path'] for r in context['inputs']])!=context['inputs'] or source_identity()!=context['code']:
            raise RuntimeError('Source or input changed during experiment; comparison invalid')
        # Generated shared caches are not user inputs in the launch manifest.
        # Recheck completed stages too, so an edit between A and B cannot pass.
        stage.verify_all()
        report=compare(results,context)
        report['stages']={p.parent.name:dict(seconds=read(p)['seconds'],worker=read(read(p)['result']['worker_report']))
                          for p in output.glob('*/complete.json')}
        write(output/'comparison.json',report)
        lines=['class,smfa_iou,pmfa_iou,delta_pp']+[f'{c},{v["smfa"]:.8f},{v["pmfa"]:.8f},{v["delta_pp"]:.6f}' for c,v in report['per_class_iou'].items()]
        (output/'per_class.csv').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (output/'comparison.md').write_text(f'# SMFA / PMFA\n\nSMFA: {100*report["smfa_miou"]:.4f}%\n\nPMFA: {100*report["pmfa_miou"]:.4f}%\n\n变化: {report["delta_pp"]:+.4f} 个百分点（{report["outcome"]}）。\n\n单次种子实验，不代表统计显著；两版分别过滤特征库。详细分类、覆盖、耗时、显存及回退统计见 comparison.json。\n',encoding='utf-8')
        write(output/'status.json',dict(status='complete',completed=results))
        print(f'[DONE] SMFA={report["smfa_miou"]*100:.4f}% PMFA={report["pmfa_miou"]*100:.4f}% delta={report["delta_pp"]:+.4f} pp\n{output}',flush=True)
    except BaseException as exc:
        write(output/'status.json',dict(status='incomplete',completed=results,error=str(exc)))
        raise
    finally:
        lock.unlink()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--version',help='Required fresh experiment version; never overwrites old runs')
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--parallel-gpus',type=int,nargs=2,metavar=('SMFA_GPU','PMFA_GPU'),
                   help='Run arms concurrently on distinct physical GPUs; shared inputs use --gpu')
    p.add_argument('--dataset-root',type=Path,default=Path('/home/zystu2025/datasets/VOC2012'))
    p.add_argument('--image-dir',type=Path,default=Path('/home/zystu2025/datasets/VOC2012/JPEGImages'))
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/sam3/sam3.pt')
    for name in ('train-list','val-list','imagelevel-json','gt-dir','pseudo-dir','proposals-dir'):
        p.add_argument('--'+name,type=Path)
    for key,value in asdict(PMFA()).items():
        p.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true',help='Read-only input/dependency/GPU checks; no model loading or output writes')
    args=p.parse_args(argv)
    if args.worker: return args
    if not args.version or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.version): p.error('Choose a safe --version')
    if args.gpu<0: p.error('--gpu must be nonnegative')
    if args.parallel_gpus and (min(args.parallel_gpus)<0 or len(set(args.parallel_gpus))!=2):
        p.error('--parallel-gpus requires two distinct nonnegative physical GPU IDs')
    try: PMFA(**{k:getattr(args,k) for k in asdict(PMFA())})
    except ValueError as exc: p.error(str(exc))
    args.train_list=args.train_list or args.dataset_root/'ImageSets/Segmentation/trainaug.txt'
    args.val_list=args.val_list or args.dataset_root/'ImageSets/Segmentation/val.txt'
    args.imagelevel_json=args.imagelevel_json or args.dataset_root/'ImageSets/ImageLevel/trainaug_imagelevel.json'
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.worker: return worker(read(args.worker))
    sys.path.insert(0,str(ROOT))
    os.chdir(ROOT)
    context=preflight(args)
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed',train_count=len(context['train_ids']),val_count=len(context['val_ids']),
                              gpu=context['gpu'],reuse=context['selected']),ensure_ascii=False,indent=2))
        return
    run_pipeline(args,context)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('[STOP] Interrupted; only owned workers terminated',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
