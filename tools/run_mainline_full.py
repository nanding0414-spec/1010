"""Fresh two-GPU LSA + HC-GMP + PRR mainline (balanced/post5/area100).

No historical experiment inputs. Immutable code snapshots, checked checkpoints,
new stage attempts, and parent-only result email. Run --help for the interface.
"""
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
RESERVE = 5 * 2**30


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), size=path.stat().st_size, sha256=sha(path))


def verify(records):
    for row in records:
        if identity(row['path']) != row:
            raise ValueError(f"Changed input/artifact: {row['path']}")


def coverage(shards, expected):
    actual = [i for shard in shards for i in shard]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError('Missing, duplicate or unexpected image IDs')


def free_space(root, required=RESERVE):
    available = shutil.disk_usage(root).free
    if available < required:
        raise OSError(f'Disk free {available / 2**30:.2f} GiB < required {required / 2**30:.2f} GiB')
    return available


def idle_gpus(gpus):
    for attempt in range(16):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        rows = {int(r.split(',')[0]): tuple(int(x.strip()) for x in r.split(',')[1:])
                for r in output.strip().splitlines()}
        if any(g not in rows or rows[g][0] > 1000 for g in gpus):
            raise RuntimeError(f'GPU unavailable/occupied: {rows}')
        if all(rows[g][1] <= 10 for g in gpus):
            return rows
        if attempt < 15:
            time.sleep(2)
    raise RuntimeError(f'GPU utilization remains high: {rows}')


def source_files(root):
    files = list(root.glob('*.py'))
    for name in ('src', 'tools', 'configs', 'project_utils', 'sam3-main'):
        files += [p for p in (root / name).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.txt', '.gz', '.toml')]
    return sorted(set(files))


def proposal_args(dataset, checkpoint, output):
    return ['--image-dir', str(Path(dataset) / 'JPEGImages'), '--checkpoint', str(checkpoint),
            '--output-dir', str(output), '--backend', 'active', '--precision', 'fp32',
            '--seed', '42', '--confidence-threshold', '0.15', '--min-mask-area', '100',
            '--min-mask-region-area', '0', '--max-mask-area-ratio', '0.95', '--max-masks', '0',
            '--bucket-size', '4', '--preview', '--preview-resolution', '672', '--presence-threshold', '0.2',
            '--active-keep-ratio', '0.4', '--min-active-classes', '3', '--max-active-classes', '12',
            '--neighbor-tokens', '2', '--hypernym-tokens', '2', '--context-strength', '1', '--nms-threshold', '0.7']


def pseudo_args(dataset, checkpoint, output, labels):
    from configs.experiment import PSEUDO, cli_options
    return cli_options(PSEUDO) + json.loads(os.environ.get('MODUSEG_PSEUDO_EXTRA_CLI', '[]')) + ['--dataset-root', str(dataset), '--ckpt', str(checkpoint),
            '--output-dir', str(output), '--image-dir', str(Path(dataset) / 'JPEGImages'),
            '--imagelevel-json', str(labels), '--no-evaluate']


def preflight(args):
    import numpy as np
    from PIL import Image
    from configs.experiment import PSEUDO, TRAINING, CLEANUP, PSEUDO_CLEANUP, MAINLINE
    from tools.proposal_demo.active_utils import load_vocabulary
    root, data, checkpoint = args.root.resolve(), args.dataset_root.resolve(), args.checkpoint.resolve()
    if (PSEUDO['dataset'], PSEUDO['split'], PSEUDO['no_bf16']) != ('voc', 'trainaug', True):
        raise ValueError('This entry requires VOC trainaug Active FP32')
    if TRAINING['recipe'] != 'v1' or TRAINING['num_layers'] != 2:
        raise ValueError('This entry requires the v1 two-layer training recipe')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainlist = data / 'ImageSets/Segmentation/trainaug.txt'
    vallist = data / 'ImageSets/Segmentation/val.txt'
    labels = data / 'ImageSets/ImageLevel/trainaug_imagelevel.json'
    train, val = trainlist.read_text().split(), vallist.read_text().split()
    coverage([train], train); coverage([val], val)
    if len(train) != 10582 or len(val) != 1449 or set(train) & set(val):
        raise ValueError('Requires disjoint full VOC trainaug=10582 and val=1449')
    records = read(labels)['images']
    coverage([[Path(r['file_name']).stem for r in records]], train)
    for row in records:
        if (data / row['file_name']).resolve() != (data / 'JPEGImages' / f"{Path(row['file_name']).stem}.jpg").resolve():
            raise ValueError('Unexpected image-level image path')
        if not isinstance(row.get('labels'), list) or any(type(v) is not int or not 0 <= v <= 20 for v in row['labels']):
            raise ValueError('Invalid training image-level labels')
    vocabulary = load_vocabulary(CODE / 'tools/proposal_demo/voc_classes.json')
    if len(vocabulary) != 20:
        raise ValueError('Requires the full VOC foreground vocabulary')
    inputs = [identity(p) for p in (checkpoint, trainlist, vallist, labels)]
    total_pixels = 0
    for pos, stem in enumerate(train + val, 1):
        image = data / 'JPEGImages' / f'{stem}.jpg'
        gt = data / 'SegmentationClassAug' / f'{stem}.png'
        with Image.open(image) as im:
            im.load(); size = im.size; total_pixels += im.width * im.height
        with Image.open(gt) as im:
            im.load(); array = np.asarray(im)
            if im.size != size or array.ndim != 2 or not np.isin(array, list(range(21)) + [255]).all():
                raise ValueError(f'Invalid GT dimensions/labels: {gt}')
        inputs += [identity(image), identity(gt)]
        if pos % 2000 == 0 or pos == len(train) + len(val):
            print(f'[Preflight] decoded/hashed {pos}/{len(train)+len(val)}', flush=True)
    # Conservative raw mask storage bound: 256 uint8 candidates per image,
    # plus clean copies, feature caches, code, diagnostics and temporary space.
    estimate = int(total_pixels / (len(train)+len(val)) * len(val) * 256 * 2 + 10 * 2**30)
    if args.save_image_features:
        estimate += (len(train)+len(val))*72*72*1280*4 + 2*2**30
    disk = free_space(root, RESERVE + estimate)
    gpu = idle_gpus(args.gpus)
    print(f'[Preflight] disk={disk/2**30:.1f} GiB; estimate={estimate/2**30:.1f} GiB; GPUs={gpu}', flush=True)
    return dict(format_version=1, version=args.version, dataset=str(data), checkpoint=str(checkpoint),
                gpus=args.gpus, seed=args.seed, train_ids=train, val_ids=val,
                inputs=inputs, code={str(p.relative_to(root)): sha(p) for p in source_files(root)},
                pseudo=PSEUDO, proposals=proposal_args(data, checkpoint, '<new raw cache>'),
                training=dict(TRAINING, seed=args.seed), cleanup=CLEANUP,
                pooling='hard_center_gmp', rho=.1, resolution=1008, estimated_bytes=estimate,
                pseudo_extra_cli=json.loads(os.environ.get('MODUSEG_PSEUDO_EXTRA_CLI','[]')),
                save_image_features=args.save_image_features,
                semantic_cleanup=args.semantic_cleanup, label_cleanup=PSEUDO_CLEANUP,
                recipe_id=MAINLINE['recipe_id'])


def environment(manifest, code, gpu=None, **overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith('OVERRIDE_')}
    env.update(PYTHONPATH=str(code) + os.pathsep + str(code / 'sam3-main'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', NVIDIA_TF32_OVERRIDE='0',
               MODUSEG_MIN_FREE_BYTES=str(RESERVE), MODUSEG_PSEUDO_CHECKPOINT_INTERVAL='1',
               MODUSEG_STRICT_CACHE='1',
               CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OVERRIDE_DATASET_TYPE='voc', OVERRIDE_VOC_TRAIN_SPLIT='trainaug',
               OVERRIDE_DATASET_ROOT=manifest['dataset'], OVERRIDE_INF_DATASET_ROOT=manifest['dataset'],
               OVERRIDE_SEED=str(manifest['seed']), OVERRIDE_SAM3_CKPT=manifest['checkpoint'],
               OVERRIDE_SAM3_POOLING='hard_center_gmp', OVERRIDE_SAM3_GMP_RHO='0.1',
               OVERRIDE_SAM3_USE_BF16='false', OVERRIDE_SAM3_RESOLUTION='1008',
               OVERRIDE_SAM3_FEATURE_SOURCE='trunk_fpn', OVERRIDE_SAM3_FEATURE_LEVEL='-1',
               OVERRIDE_SAM3_TOKEN_NORM='layernorm', OVERRIDE_SAM3_CONTEXT_SUBTRACT='0',
               OVERRIDE_FEATURE_ADAPTER_CKPT='', OVERRIDE_MIN_MASK_AREA='100',
               OVERRIDE_BUILD_SPLIT_CONNECTED_COMPONENTS='true',
               OVERRIDE_FILTER_FOREGROUND_DROP_RATIO='0', OVERRIDE_FILTER_BACKGROUND_DROP_RATIO='0',
               OVERRIDE_INF_RETRIEVAL_MODE='balanced', OVERRIDE_INF_TOPK_NEIGH='25',
               OVERRIDE_INF_CLASS_TOPM='5', OVERRIDE_INF_RETRIEVAL_SEARCH_MULTIPLIER='2',
               OVERRIDE_INF_CENTROID_WEIGHT='0.2', OVERRIDE_INF_RETRIEVAL_MARGIN_THR='0',
               OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT='0.2', OVERRIDE_INF_NMS_IOU='0.4',
               OVERRIDE_INF_INSTANCE_SCORE_THR='0.02', OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK='false',
               OVERRIDE_INF_VIS_LIMIT='20')
    env.update({k: str(v) for k, v in overrides.items()})
    env['MODUSEG_PSEUDO_EXTRA_CLI'] = json.dumps(manifest.get('pseudo_extra_cli', []))
    return env


def run_script(script, argv):
    import runpy
    old = sys.argv
    sys.argv = [str(CODE / script), *map(str, argv)]
    try:
        try:
            runpy.run_path(str(CODE / script), run_name='__main__')
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(f'{script} exited {error.code}') from error
    finally:
        sys.argv = old
        gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            sys.modules['torch'].cuda.empty_cache()


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def checked_checkpoint(path, binding):
    path = Path(path); receipt = path.with_suffix('.json')
    if not path.exists() and not receipt.exists():
        return False
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f'Incomplete checkpoint pair retained: {path}')
    record = read(receipt)
    if record['binding'] != binding or record['artifact'] != identity(path):
        raise ValueError(f'Checkpoint identity differs: {path}')
    return True


def build_images(job):
    import cv2
    import numpy as np
    import torch
    from src.feature_extractor_factory import build_feature_extractor
    from src.region_extractors import get_regions_from_pseudo
    model = build_feature_extractor(torch.device('cuda:0'))
    signature = model.feature_signature()
    write(job['signature'], signature)
    for number, stem in enumerate(job['ids'], 1):
        free_space(job['root'])
        image = Path(job['dataset']) / 'JPEGImages' / f'{stem}.jpg'
        mask = Path(job['pseudo']) / f'{stem}.png'
        binding = dict(image=identity(image), mask=identity(mask), signature=signature, code=job['code_hash'])
        if getattr(model, 'image_token_cache', None) is not None:
            rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
            model.set_image_source(image, rgb)
            model.cache_image_features(rgb)
        target = Path(job['cache']) / f'{stem}.npz'
        if not checked_checkpoint(target, binding):
            regions = get_regions_from_pseudo(str(image), 300, job['pseudo'])
            rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
            vectors = model.extract_features_batch(rgb, [r['mask'] for r in regions])
            if len(vectors) != len(regions):
                raise ValueError(f'Incomplete feature extraction: {stem}')
            x = np.asarray(vectors, np.float32).reshape(len(regions), model.embed_dim)
            if not np.isfinite(x).all() or (len(x) and (np.linalg.norm(x, axis=1) < 1e-6).any()):
                raise ValueError(f'Invalid region feature: {stem}')
            save_npz(target, features=x, class_ids=np.array([r['cls_id'] for r in regions], np.int16),
                     sample_ids=np.array([f'{stem}_{i:06d}' for i in range(len(regions))]))
            write(target.with_suffix('.json'), dict(binding=binding, artifact=identity(target), regions=len(regions)))
        print(f'[Build] {number}/{len(job["ids"])} {stem}', flush=True)


def merge_bank(cache, ids, signatures, target, filter_members=True):
    import numpy as np
    from configs import config as cfg
    from src.feature_bank_storage import save_feature_matrix_npz
    from src.feature_signature import save_feature_signature
    from src.build_feature_bank import save_class_mapping
    signature = read(signatures[0])
    if any(read(s) != signature for s in signatures):
        raise ValueError('GPU feature signatures differ')
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    rows = {i: [] for i in range(21)}; names = {i: [] for i in range(21)}
    coverage([[p.stem for p in Path(cache).glob('*.npz')]], ids)
    for stem in sorted(ids):
        with np.load(Path(cache) / f'{stem}.npz', allow_pickle=False) as values:
            x, cids, samples = values['features'], values['class_ids'], values['sample_ids']
        if x.shape != (len(cids), signature['embed_dim']) or len(samples) != len(cids):
            raise ValueError('Wrong checkpoint feature dimensions')
        if not np.isfinite(x).all() or (len(x) and (np.linalg.norm(x, axis=1) < 1e-6).any()):
            raise ValueError('Invalid checkpoint vectors')
        for feature, cid, sample in zip(x, cids, samples):
            if int(cid) not in rows:
                raise ValueError('Invalid region class')
            rows[int(cid)].append(feature); names[int(cid)].append(str(sample))
    target.mkdir(parents=True)
    (target / 'index').mkdir()
    save_class_mapping(21, cfg.CLS_PATH)
    save_feature_signature(target / 'meta', signature)
    for cid, features in rows.items():
        if not features:
            if filter_members:
                raise ValueError(f'No training features for class {cid}')
            continue
        if len(set(names[cid])) != len(names[cid]):
            raise ValueError('Duplicate region IDs')
        name = 'background' if cid == 0 else cfg.DATASET_CLASSES[cid - 1]
        save_feature_matrix_npz(target / 'features' / name / f'{name}_features.npz', np.stack(features), names[cid])
    write(target / 'meta/coverage.json', dict(ids=ids, regions=sum(map(len, names.values())),
                                             row_order='image_id_then_region_index', historical_inputs=False))
    if filter_members:
        from src.filter_stage import run_filter_stage
        run_filter_stage()


def merge_metrics(results, expected):
    import numpy as np
    coverage([r['evaluated_ids'] for r in results], expected)
    common = ('feature_signature', 'feature_bank', 'proposals_dir', 'inference_settings', 'cleanup')
    for key in common:
        if any(r[key] != results[0][key] for r in results):
            raise ValueError(f'Evaluation settings differ: {key}')
    matrices = [np.asarray(r['confusion_matrix']) for r in results]
    if any(m.shape != (21, 21) or not np.issubdtype(m.dtype, np.integer) or (m < 0).any() for m in matrices):
        raise ValueError('Invalid VOC confusion matrix')
    matrix = sum(matrices, np.zeros((21, 21), np.int64))
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.zeros(21, float), where=union > 0)
    names = list(results[0]['per_class_iou'])
    if len(names) != 21 or any(list(r['per_class_iou']) != names for r in results):
        raise ValueError('VOC class order differs')
    merged = dict(results[0], confusion_matrix=matrix.tolist(), mIoU=float(iou.mean()),
                  per_class_iou=dict(zip(names, iou.tolist())), evaluated_ids=expected,
                  evaluated_count=len(expected), shard_index=None, shard_count=len(results),
                  aggregation='sum_confusion_then_iou')
    for name in ('proposal_count_before', 'proposal_count_after'):
        merged[name] = sum(r.get(name, 0) for r in results)
    if 'semantic_rescue' in merged:
        descriptions = [{k: v for k, v in r['semantic_rescue'].items()
                         if k not in ('images', 'accepted', 'added_pixels')} for r in results]
        if any(d != descriptions[0] for d in descriptions):
            raise ValueError('HCR settings differ')
        merged['semantic_rescue'] = dict(descriptions[0],
            images=[i for r in results for i in r['semantic_rescue']['images']],
            accepted=sum(r['semantic_rescue']['accepted'] for r in results),
            added_pixels=sum(r['semantic_rescue']['added_pixels'] for r in results))
    return merged


def probe(job):
    import cv2
    import numpy as np
    import torch
    from src.feature_extractor_factory import build_feature_extractor
    from configs.experiment import TRAINING
    from src.feature_bank_storage import save_feature_matrix_npz
    from src.feature_signature import save_feature_signature
    from src.build_feature_bank import save_class_mapping
    folder = Path(job['folder']); folder.mkdir(parents=True)
    dataset, checkpoint = Path(job['dataset']), Path(job['checkpoint'])
    labels = dict(read(dataset / 'ImageSets/ImageLevel/trainaug_imagelevel.json'))
    labels['images'] = [r for r in labels['images'] if Path(r['file_name']).stem == job['train_ids'][0]]
    write(folder / 'labels.json', labels)
    raw_pseudo = job.get('pseudo_raw', job['pseudo'])
    run_script('tools/generate_pseudo_masks_sam3.py', pseudo_args(dataset, checkpoint, raw_pseudo, folder / 'labels.json')
               + ['--device', 'cuda:0', '--experiment-dir', str(folder / 'pseudo')])
    if job.get('semantic_cleanup'):
        label_list = folder/'label_ids.txt'; label_list.write_text(job['train_ids'][0]+'\n')
        run_script('tools/clean_pseudo_labels.py', ['--input-dir', raw_pseudo, '--output-dir', job['pseudo'],
                    '--image-list', str(label_list)])
    run_script('run_generate_proposals_active.py', proposal_args(dataset, checkpoint, job['raw'])
               + ['--image-ids', job['val_id'], '--device', 'cuda:0'])
    run_script('tools/refine_proposals.py', ['--input-dir', job['raw'], '--output-dir', job['clean']])
    model = build_feature_extractor(torch.device('cuda:0'))
    signature = model.feature_signature()
    if job.get('semantic_cleanup'):
        from src.region_extractors import get_regions_from_pseudo
        path = dataset/'JPEGImages'/f'{job["train_ids"][0]}.jpg'
        rgb = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        model.set_image_source(path,rgb)
        regions = get_regions_from_pseudo(str(path),300,job['pseudo'])
        checked = model.extract_features_batch(rgb,[r['mask'] for r in regions])
        if not checked or any(not np.isfinite(x).all() or np.linalg.norm(x)<1e-6 for x in checked):
            raise ValueError('Cleaned-label GMP probe failed')
    bank = Path(job['bank']); bank.mkdir(parents=True)
    save_class_mapping(21, 'configs/cls_voc21.txt')
    save_feature_signature(bank / 'meta', signature)
    vectors = [[], []]; samples = [[], []]
    # Synthetic diagnostic class assignments exercise the training interface;
    # these vectors and their diagnostic metrics never enter the real experiment.
    for stem in job['train_ids']:
        rgb = cv2.cvtColor(cv2.imread(str(dataset / 'JPEGImages' / f'{stem}.jpg')), cv2.COLOR_BGR2RGB)
        full = np.ones(rgb.shape[:2], bool); half = full.copy(); half[:, half.shape[1] // 2:] = False
        model.set_image_source(dataset/'JPEGImages'/f'{stem}.jpg', rgb)
        result = model.extract_features_batch(rgb, [full, half])
        if getattr(model, 'image_token_cache', None) is not None:
            model.image_token_cache.live = None
            cached = model.extract_features_batch(rgb, [full, half])
            if any(not np.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(result,cached)):
                raise ValueError('Cached/live GMP vectors differ')
        for cid, vector in enumerate(result):
            if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
                raise ValueError('Probe vector invalid')
            vectors[cid].append(vector); samples[cid].append(f'{stem}_{cid:06d}')
    for cid, name in enumerate(('background', 'aeroplane')):
        save_feature_matrix_npz(bank / 'features' / name / f'{name}_features.npz', np.stack(vectors[cid]), samples[cid])
    del model; gc.collect(); torch.cuda.empty_cache()
    trainlist = folder / 'train.txt'; trainlist.write_text('\n'.join(job['train_ids']) + '\n')
    settings = dict(TRAINING, epochs=1, steps_per_epoch=2, classes_per_batch=2, instances_per_class=2, val_fraction=.25)
    from project_utils.experiment import train_command
    command = train_command(bank, job['weights'], job['mlp_bank'], trainlist, None, settings)
    run_script('tools/train_feature_adapter.py', command[3:])
    os.environ.update(OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(job['weights']) / 'adapter.pt'),
                      OVERRIDE_INF_FEATURE_BANK_ROOT=job['mlp_bank'], OVERRIDE_INF_LOAD_PROPOSALS='true',
                      OVERRIDE_INF_PROPOSALS_DIR=job['clean'])
    # Configuration was already imported by the feature/short-training probe.
    # Run inference in a fresh process so its adapter/cache overrides take effect.
    env = os.environ.copy()
    evaluation = dict(operation='script', script='run_inference.py', gpu=job['gpu'],
        report=str(folder/'eval_worker_report.json'), argv=['--strict-eval', '--eval-limit', '1',
            '--rescue', '--rescue-raw-dir', job['raw'], '--output-dir', str(folder/'eval'),
            '--results-json', str(folder/'eval/metrics.json')])
    write(folder/'eval_job.json', evaluation)
    command = [sys.executable, '-B', str(CODE/'tools/run_mainline_full.py'), '--worker-job', str(folder/'eval_job.json')]
    subprocess.run(command, env=env, check=True)
    result = read(folder / 'eval/metrics.json')
    if result['evaluated_count'] != 1 or len(read(Path(job['weights']) / 'training_report.json')['epochs']) != 1:
        raise ValueError('Short pipeline incomplete')
    write(folder / 'probe.json', dict(status='complete', feature_signature=signature,
        checked_images=job['train_ids'], proposal_image=job['val_id'], diagnostic_only=True,
        note='Synthetic class assignments; no reported experimental accuracy'))


def worker(job):
    import torch
    os.chdir(CODE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if job.get('gpu') is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Worker requires exactly one visible GPU')
        torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); status = 'failed'
    try:
        if job['operation'] == 'build':
            build_images(job)
        elif job['operation'] == 'merge':
            merge_bank(job['cache'], job['ids'], job['signatures'], job['bank'])
        elif job['operation'] == 'postfilter':
            from src.semantic_bank_filter import export_filtered_bank
            export_filtered_bank(job['source_bank'], job['target_bank'])
        elif job['operation'] == 'probe':
            probe(job)
        else:
            run_script(job['script'], job['argv'])
        status = 'complete'
    finally:
        child_peak = 0
        if job['operation'] == 'probe' and (Path(job['folder'])/'eval_worker_report.json').is_file():
            child_peak = read(Path(job['folder'])/'eval_worker_report.json')['peak_allocated_mib']
        write(job['report'], dict(status=status, operation=job['operation'], physical_gpu=job.get('gpu'),
              seconds=time.monotonic() - started,
              peak_allocated_mib=max(child_peak, torch.cuda.max_memory_allocated()/2**20) if job.get('gpu') is not None else 0,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if job.get('gpu') is not None else 0))


def files_under(paths):
    return sorted({p for path in paths for p in ([Path(path)] if Path(path).is_file()
                  else Path(path).rglob('*')) if p.is_file() and not p.name.endswith('.tmp')})


class Launcher:
    def __init__(self, args, manifest, folder, code):
        self.args, self.manifest, self.folder, self.code = args, manifest, folder, code
        self.current = 'initialization'; self.workers = []; self.reports = []
        self.logroot = args.root / 'logs' / args.version

    def jobs(self, stage, attempt, jobs):
        free_space(self.args.root)
        used = [j.get('gpu') for j in jobs if j.get('gpu') is not None]
        if len(set(used)) != len(used):
            raise ValueError('At most one worker per GPU')
        if used:
            idle_gpus(used)
        streams = []; self.workers = []
        try:
            for index, job in enumerate(jobs):
                job = dict(job, report=str(attempt / f'worker_{index}_report.json'))
                spec = attempt / f'job_{index}.json'; write(spec, job)
                log = self.logroot / stage / attempt.name / f'worker_{index}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('x', encoding='utf-8'); streams.append(stream)
                env = environment(self.manifest, self.code, job.get('gpu'), **job.get('overrides', {}))
                command = [sys.executable, '-u', '-B', str(self.code / 'tools/run_mainline_full.py'), '--worker-job', str(spec)]
                process = subprocess.Popen(command, env=env, cwd=self.code, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.workers.append(process)
                write(attempt / f'worker_{index}_process.json', dict(pid=process.pid, command=command, log=str(log),
                      physical_gpu=job.get('gpu'), environment={k: v for k, v in env.items()
                        if k.startswith('OVERRIDE_') or k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TMPDIR')}))
                print(f'[Worker] {stage} GPU={job.get("gpu")} PID={process.pid} log={log}', flush=True)
            write(self.folder / 'progress.json', dict(stage=stage, attempt=attempt.name,
                  worker_pids=[p.pid for p in self.workers], updated=time.time()))
            while any(p.poll() is None for p in self.workers):
                if any(p.poll() not in (None, 0) for p in self.workers):
                    raise RuntimeError(f'{stage} worker failed; inspect {self.logroot / stage / attempt.name}')
                time.sleep(1)
            if any(p.returncode != 0 for p in self.workers):
                raise RuntimeError(f'{stage} worker failed')
            for index in range(len(jobs)):
                self.reports.append(dict(stage=stage, **read(attempt / f'worker_{index}_report.json')))
        finally:
            for process in self.workers:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in self.workers:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            for stream in streams:
                stream.close()
            self.workers = []

    def stage(self, name, action):
        self.current = name
        verify(self.manifest['inputs'])
        directory = self.folder / name; marker = directory / 'complete.json'
        if marker.is_file():
            value = read(marker); verify(value['artifacts'])
            self.reports += value.get('workers', [])
            print(f'[Resume] verified {name}', flush=True)
            return value['result']
        directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while (directory / f'attempt_{index:03d}').exists():
            index += 1
        attempt = directory / f'attempt_{index:03d}'; attempt.mkdir()
        before = len(self.reports)
        write(self.folder / 'status.json', dict(status='running', stage=name, attempt=attempt.name))
        result, paths = action(attempt)
        write(marker, dict(result=result, artifacts=[identity(p) for p in files_under(paths)], workers=self.reports[before:]))
        print(f'[Complete] {name}', flush=True)
        return result


def validate_pseudo(folder, ids, dataset):
    import numpy as np
    from PIL import Image
    coverage([[p.stem for p in Path(folder).glob('*.png')]], ids)
    for stem in ids:
        with Image.open(Path(folder)/f'{stem}.png') as mask, Image.open(Path(dataset)/'JPEGImages'/f'{stem}.jpg') as image:
            arr = np.asarray(mask)
            if mask.size != image.size or arr.ndim != 2 or not np.isin(arr, range(21)).all():
                raise ValueError(f'Invalid pseudo label: {stem}')


def validate_proposals(folder, ids, dataset, clean=False):
    import numpy as np
    from PIL import Image
    folder = Path(folder)
    coverage([[p.stem for p in folder.glob('*.npz')]], ids)
    for stem in ids:
        with Image.open(Path(dataset)/'JPEGImages'/f'{stem}.jpg') as image:
            shape = (image.height, image.width)
        with np.load(folder/f'{stem}.npz', allow_pickle=False) as data:
            masks, scores = data['masks'], data['scores']
            if (masks.shape != (len(scores), *shape) or scores.shape != (len(masks),)
                    or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()
                    or not np.isin(masks, [0, 1]).all()):
                raise ValueError(f'Invalid proposal cache: {stem}')
            if clean and data['source_indices'].shape != scores.shape:
                raise ValueError('Invalid clean source indices')
    if clean:
        meta = read(folder/'cleanup_config.json')
        if meta['errors'] != 0 or meta['total'] != len(ids):
            raise ValueError('Incomplete cleanup')
        for record in meta['files']:
            if sha(Path(meta['input_dir'])/record['name']) != record['source_sha256']:
                raise ValueError('Raw/clean provenance differs')


def pipeline(launch, probe_only=False):
    m, root, folder = launch.manifest, launch.args.root, launch.folder
    version = m['version']; dataset = Path(m['dataset']); checkpoint = Path(m['checkpoint'])
    labelbase = root/'storage/pseudo_labels/voc'/version
    pseudo = labelbase/'raw' if m.get('semantic_cleanup') else labelbase
    proposals = root/'storage/proposals/voc'/version
    rawbase = root/'storage/raw_banks/voc'/version
    raw = proposals/'raw'; cache = rawbase/'image_cache'
    weightsbase = root/'weights/sam3_adapter'/version
    mlpbase = root/'storage/mlp_banks/voc'/version
    imagefeatures = root/'storage/image_features/voc'/version
    own = [labelbase, proposals, rawbase, weightsbase, mlpbase, launch.logroot, folder,
           root/'storage/runtime'/version]
    write(folder/'artifacts.json', dict(owned_roots=list(map(str, own)), historical_inputs=False))
    if m.get('save_image_features'):
        write(folder/'artifacts.json', dict(owned_roots=list(map(str,own+[imagefeatures])),historical_inputs=False))
    labels_data = read(dataset/'ImageSets/ImageLevel/trainaug_imagelevel.json')
    byid = {Path(r['file_name']).stem: r for r in labels_data['images']}
    def do_probe(attempt):
        jobs = []
        probe_ids = m['train_ids'][:4]
        if '--edge-repair' in m.get('pseudo_extra_cli', []):
            table_image = next(i for i in m['train_ids'] if 11 in byid[i]['labels'])
            probe_ids = [table_image] + [i for i in m['train_ids'] if i != table_image][:3]
        for gpu in m['gpus']:
            suffix = f'probe/{attempt.name}/gpu{gpu}'
            jobs.append(dict(operation='probe', gpu=gpu, dataset=str(dataset), checkpoint=str(checkpoint),
                folder=str(attempt/f'gpu{gpu}'), train_ids=probe_ids, val_id=m['val_ids'][0],
                pseudo=str(labelbase/suffix/'clean') if m.get('semantic_cleanup') else str(labelbase/suffix),
                **({'pseudo_raw':str(labelbase/suffix/'raw'),'semantic_cleanup':True} if m.get('semantic_cleanup') else {}),
                raw=str(proposals/suffix/'raw'), clean=str(proposals/suffix/'clean'),
                bank=str(rawbase/suffix/'bank'), weights=str(weightsbase/suffix), mlp_bank=str(mlpbase/suffix),
                overrides=dict({'OVERRIDE_OUTPUT_ROOT': str(rawbase/suffix/'bank')},
                    **({'OVERRIDE_IMAGE_TOKEN_CACHE_DIR': str(imagefeatures/suffix)} if m.get('save_image_features') else {}))))
        launch.jobs('probe', attempt, jobs)
        return dict(status='complete', diagnostic_only=True), [attempt] + [p/'probe'/attempt.name for p in own[:5]] + (
            [imagefeatures/'probe'/attempt.name] if m.get('save_image_features') else [])
    launch.stage('probe', do_probe)
    if probe_only:
        return dict(status='ready', version=version, diagnostic_only=True, workers=launch.reports,
                    note='Both GPU real probes and short pipelines passed; full dataset generation not started')
    def do_pseudo(attempt):
        jobs = []
        shards = [m['train_ids'][i::2] for i in range(2)]; coverage(shards, m['train_ids'])
        for index, gpu in enumerate(m['gpus']):
            source = attempt/f'shard_{index}.json'
            write(source, dict(labels_data, images=[byid[i] for i in shards[index]]))
            jobs.append(dict(operation='script', script='tools/generate_pseudo_masks_sam3.py', gpu=gpu,
                argv=pseudo_args(dataset, checkpoint, pseudo, source) + ['--device', 'cuda:0',
                    '--metadata-dir', str(pseudo/'meta'/f'shard_{index}'), '--experiment-dir', str(attempt/f'gpu{gpu}')]))
        launch.jobs('pseudo', attempt, jobs)
        validate_pseudo(pseudo, m['train_ids'], dataset)
        return dict(path=str(pseudo), count=len(m['train_ids'])), [p for p in pseudo.glob('*.png')] + [pseudo/'meta', attempt]
    launch.stage('pseudo', do_pseudo)
    label_cleanup = None
    if m.get('semantic_cleanup'):
        def do_label_cleanup(attempt):
            cleaned = labelbase/'clean'/attempt.name
            launch.jobs('pseudo_cleanup',attempt,[dict(operation='script',script='tools/clean_pseudo_labels.py',
                argv=['--input-dir',str(pseudo),'--output-dir',str(cleaned),
                      '--image-list',str(dataset/'ImageSets/Segmentation/trainaug.txt')])])
            validate_pseudo(cleaned,m['train_ids'],dataset)
            metadata=read(cleaned/'cleanup_config.json')
            if metadata['status']!='complete' or metadata['count']!=len(m['train_ids']):
                raise ValueError('Semantic label cleanup incomplete')
            return dict(path=str(cleaned),statistics=metadata['statistics']),[cleaned,attempt]
        label_cleanup = launch.stage('pseudo_cleanup',do_label_cleanup)
        pseudo = Path(label_cleanup['path'])
    def do_pseudo_eval(attempt):
        output = attempt/'evaluation'
        args = pseudo_args(dataset, checkpoint, pseudo, dataset/'ImageSets/ImageLevel/trainaug_imagelevel.json')
        launch.jobs('pseudo_eval', attempt, [dict(operation='script', script='tools/generate_pseudo_masks_sam3.py',
            argv=args+['--eval-only', '--evaluate', '--device', 'cpu', '--gt-dir', str(dataset/'SegmentationClassAug'),
                       '--experiment-dir', str(output)])])
        report = read(output/'miou_trainaug.json')
        if report['status'] != 'complete' or report['evaluated_count'] != len(m['train_ids']):
            raise ValueError('Pseudo evaluation incomplete')
        return dict(metrics=str(output/'miou_trainaug.json'), mIoU=report['mIoU']), [output, pseudo/'meta']
    pseudo_quality = launch.stage('pseudo_eval', do_pseudo_eval)
    def do_proposals(attempt):
        from run_sam3_proposal_demo import parse_args
        from run_generate_proposals_active import prepare_output
        common = proposal_args(dataset, checkpoint, raw)
        prepare_output(parse_args(common+['--split', str(dataset/'ImageSets/Segmentation/val.txt')], bulk=True))
        jobs = []
        for index, gpu in enumerate(m['gpus']):
            ids = m['val_ids'][index::2]; split = attempt/f'shard_{index}.txt'
            split.write_text('\n'.join(ids)+'\n')
            jobs.append(dict(operation='script', script='run_generate_proposals_active.py', gpu=gpu,
                             argv=common+['--split', str(split), '--device', 'cuda:0']))
        launch.jobs('proposals', attempt, jobs)
        validate_proposals(raw, m['val_ids'], dataset)
        return dict(path=str(raw), count=len(m['val_ids'])), [raw, attempt]
    launch.stage('proposals', do_proposals)
    def do_cleanup(attempt):
        clean = proposals/'clean'/attempt.name
        launch.jobs('cleanup', attempt, [dict(operation='script', script='tools/refine_proposals.py',
                     argv=['--input-dir', str(raw), '--output-dir', str(clean)])])
        validate_proposals(clean, m['val_ids'], dataset, clean=True)
        return dict(path=str(clean)), [clean, attempt]
    clean = Path(launch.stage('cleanup', do_cleanup)['path'])
    def do_build(attempt):
        jobs = []
        for index, gpu in enumerate(m['gpus']):
            jobs.append(dict(operation='build', gpu=gpu, root=str(root), dataset=str(dataset), pseudo=str(pseudo),
                ids=m['train_ids'][index::2], cache=str(cache), signature=str(attempt/f'signature_{index}.json'),
                code_hash=sha(folder/'manifest.json'), overrides=dict({'OVERRIDE_PSEUDO_MASK_ROOT': str(pseudo)},
                    **({'OVERRIDE_IMAGE_TOKEN_CACHE_DIR': str(imagefeatures/'train')} if m.get('save_image_features') else {}))))
        launch.jobs('build', attempt, jobs)
        coverage([[p.stem for p in cache.glob('*.npz')]], m['train_ids'])
        if m.get('save_image_features'):
            coverage([[p.stem for p in (imagefeatures/'train').glob('*.npz')]], m['train_ids'])
        return dict(cache=str(cache), signatures=[str(attempt/f'signature_{i}.json') for i in range(2)]), [cache, attempt] + (
            [imagefeatures/'train'] if m.get('save_image_features') else [])
    extracted = launch.stage('build', do_build)
    def do_merge(attempt):
        bank = rawbase/'bank'/attempt.name
        launch.jobs('merge', attempt, [dict(operation='merge', cache=str(cache), ids=m['train_ids'],
            signatures=extracted['signatures'], bank=str(bank), overrides={'OVERRIDE_OUTPUT_ROOT': str(bank)})])
        return dict(path=str(bank)), [bank, attempt]
    bank = Path(launch.stage('merge', do_merge)['path'])
    def do_train(attempt):
        from project_utils.experiment import train_command
        weights = weightsbase/str(m['seed'])/attempt.name; mlp = mlpbase/str(m['seed'])/'unfiltered'/attempt.name
        command = train_command(bank, weights, mlp, dataset/'ImageSets/Segmentation/trainaug.txt', None, m['training'])
        launch.jobs('train', attempt, [dict(operation='script', script='tools/train_feature_adapter.py',
            argv=command[3:], gpu=m['gpus'][1])])
        report = read(weights/'training_report.json')
        if len(report['epochs']) != 50 or report['args']['num_layers'] != 2:
            raise ValueError('Incomplete MLP2 training')
        return dict(weights=str(weights), bank=str(mlp), best_epoch=report['best_epoch']), [weights, mlp, attempt]
    trained = launch.stage('train', do_train)
    def do_postfilter(attempt):
        target = mlpbase/str(m['seed'])/'post05'/attempt.name
        launch.jobs('postfilter', attempt, [dict(operation='postfilter',
            source_bank=trained['bank'], target_bank=str(target))])
        return dict(path=str(target)), [target, attempt]
    filtered_bank = launch.stage('postfilter', do_postfilter)['path']
    def do_eval(attempt):
        jobs = []
        for index, gpu in enumerate(m['gpus']):
            out = attempt/f'gpu{gpu}'
            jobs.append(dict(operation='script', script='run_inference.py', gpu=gpu,
                argv=['--strict-eval', '--eval-limit', '0', '--shard-count', '2', '--shard-index', str(index),
                      '--rescue', '--prr', '--rescue-raw-dir', str(raw), '--output-dir', str(out), '--results-json', str(out/'metrics.json')],
                overrides={'OVERRIDE_FEATURE_ADAPTER_CKPT': str(Path(trained['weights'])/'adapter.pt'),
                           'OVERRIDE_INF_FEATURE_BANK_ROOT': filtered_bank, 'OVERRIDE_INF_LOAD_PROPOSALS': 'true',
                           'OVERRIDE_INF_PROPOSALS_DIR': str(clean),
                           **({'OVERRIDE_IMAGE_TOKEN_CACHE_DIR': str(imagefeatures/'val')} if m.get('save_image_features') else {})}))
        launch.jobs('eval', attempt, jobs)
        metrics = merge_metrics([read(attempt/f'gpu{g}'/'metrics.json') for g in m['gpus']], m['val_ids'])
        if m.get('save_image_features'):
            coverage([[p.stem for p in (imagefeatures/'val').glob('*.npz')]], m['val_ids'])
        write(attempt/'metrics.json', metrics)
        with (attempt/'per_class.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['class', 'IoU', 'IoU_percent'])
            for name, value in metrics['per_class_iou'].items():
                writer.writerow([name, value, value*100])
        return dict(metrics=str(attempt/'metrics.json'), per_class=str(attempt/'per_class.csv')), [attempt] + (
            [imagefeatures/'val'] if m.get('save_image_features') else [])
    evaluated = launch.stage('eval', do_eval)
    for source, target in ((evaluated['metrics'], folder/'metrics.json'), (evaluated['per_class'], folder/'per_class.csv')):
        if target.exists():
            if sha(target) != sha(source):
                raise ValueError(f'Refusing to overwrite previously published metrics: {target}')
        else:
            shutil.copy2(source, target)
    metrics = read(folder/'metrics.json')
    summary = dict(status='complete', version=version, seed=m['seed'], mIoU=metrics['mIoU'],
        per_class_iou=metrics['per_class_iou'], evaluated_count=metrics['evaluated_count'],
        train_count=len(m['train_ids']), pseudo_quality=pseudo_quality, best_epoch=trained['best_epoch'],
        region_statistics=read(bank/'meta/filter_report.json'), workers=launch.reports,
        raw_proposals=str(raw), clean_proposals=str(clean), raw_bank=str(bank),
        mlp_bank=filtered_bank, unfiltered_mlp_bank=trained['bank'], weights=trained['weights'], historical_inputs=False,
        pseudo_extra_cli=m.get('pseudo_extra_cli',[]),
        full_image_features=str(imagefeatures) if m.get('save_image_features') else None)
    summary.update(recipe_id=m.get('recipe_id'),pseudo_labels=str(pseudo),
                   raw_pseudo_labels=str(labelbase/'raw') if m.get('semantic_cleanup') else str(labelbase),
                   semantic_cleanup=label_cleanup)
    return summary


def parse_args(argv=None):
    from configs.experiment import PSEUDO, MAINLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=CODE)
    parser.add_argument('--version')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    parser.add_argument('--seed', type=int, choices=[42,123,2026], default=42)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--probe-only', action='store_true', help='Run both real GPU/short-pipeline checks, then stop before full generation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--edge-repair', action=argparse.BooleanOptionalAction,default=PSEUDO['edge_repair'])
    parser.add_argument('--force-presence-one', action=argparse.BooleanOptionalAction,default=PSEUDO['force_presence_one'])
    parser.add_argument('--mask-logit-threshold', type=float, default=.4)
    parser.add_argument('--close-kernel', type=int, default=3)
    parser.add_argument('--close-iterations', type=int, default=1)
    parser.add_argument('--dilate-pixels', type=int, default=1)
    parser.add_argument('--diningtable-close-iterations', type=int, default=2)
    parser.add_argument('--save-image-features', action=argparse.BooleanOptionalAction,default=MAINLINE['save_image_features'])
    parser.add_argument('--semantic-cleanup', action=argparse.BooleanOptionalAction,default=MAINLINE['semantic_cleanup'])
    parser.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.worker_job:
        worker(read(args.worker_job)); return
    args.root = args.root.resolve()
    if not args.version or not all(c.isalnum() or c in '_-' for c in args.version) or len(set(args.gpus)) != 2:
        raise ValueError('Provide a safe new --version and two distinct physical GPUs')
    args.dataset_root = (args.dataset_root or args.root/'data/VOC2012').resolve()
    args.checkpoint = (args.checkpoint or args.root/'weights/sam3/sam3.pt').resolve()
    extra = ['--edge-repair' if args.edge_repair else '--no-edge-repair',
             '--force-presence-one' if args.force_presence_one else '--no-force-presence-one']
    if args.edge_repair:
        from src.mask_edge_repair import sam3_logits_to_repaired_mask
        import numpy as np
        sam3_logits_to_repaired_mask(np.zeros((1,1)),args.mask_logit_threshold,args.close_kernel,args.close_iterations,args.dilate_pixels)
        if args.diningtable_close_iterations < 0:
            raise ValueError('Invalid diningtable iterations')
        for name in ('mask_logit_threshold','close_kernel','close_iterations','dilate_pixels','diningtable_close_iterations'):
            extra += ['--'+name.replace('_','-'),str(getattr(args,name))]
    os.environ['MODUSEG_PSEUDO_EXTRA_CLI'] = json.dumps(extra)
    # Do not import configs under ambient experiment overrides.
    for key in list(os.environ):
        if key.startswith('OVERRIDE_'):
            del os.environ[key]
    folder = args.root/'experiments'/args.version
    if folder.exists() and not args.resume:
        raise FileExistsError(f'Choose a new experiment version: {folder}')
    if args.resume and not (folder/'manifest.json').is_file():
        raise ValueError('--resume requires an existing manifest')
    manifest = preflight(args)
    if args.preflight_only:
        print('[Preflight] passed; no experiment started', flush=True); return
    if args.resume:
        if read(folder/'manifest.json') != manifest:
            raise ValueError('Code, inputs or recipe changed; use a new version')
    else:
        for parent in ('storage/pseudo_labels/voc', 'storage/proposals/voc', 'storage/raw_banks/voc',
                       'storage/mlp_banks/voc', 'storage/image_features/voc', 'weights/sam3_adapter', 'logs'):
            if (args.root/parent/args.version).exists():
                raise FileExistsError(args.root/parent/args.version)
        folder.mkdir(parents=True); write(folder/'manifest.json', manifest)
    lock = folder/'launcher.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started=time.time())))
    code = folder/'code'
    started = time.monotonic(); launch = None; summary = None
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    old_signal = signal.signal(signal.SIGTERM, interrupted)
    try:
        if not code.exists():
            code.mkdir()
            for relative in manifest['code']:
                target = code/relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.root/relative, target)
        if any(sha(code/p) != digest for p, digest in manifest['code'].items()):
            raise ValueError('Immutable code snapshot differs')
        sys.path[:0] = [str(code), str(code/'sam3-main')]
        runtime = args.root/'storage/runtime'/args.version
        for name in ('tmp', 'huggingface', 'torch', 'xdg'):
            (runtime/name).mkdir(parents=True, exist_ok=True)
        os.environ.update(TMPDIR=str(runtime/'tmp'), TEMP=str(runtime/'tmp'), TMP=str(runtime/'tmp'),
                          HF_HOME=str(runtime/'huggingface'), TORCH_HOME=str(runtime/'torch'),
                          XDG_CACHE_HOME=str(runtime/'xdg'))
        launch = Launcher(args, manifest, folder, code)
        summary = pipeline(launch, args.probe_only)
    except BaseException as error:
        summary = dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', version=args.version,
            failed_stage=launch.current if launch else 'initialization', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        if summary is None:
            summary = dict(status='failed', version=args.version, failed_stage='initialization')
        summary['seconds'] = time.monotonic() - started
        write(folder/'summary.json', summary)
        write(folder/'status.json', dict(status=summary['status'], stage=summary.get('failed_stage', 'complete'),
              seconds=summary['seconds'], finished=time.time()))
        if summary['status'] != 'ready':
            from project_utils.experiment_notify import notify_experiment
            attachments = [p for p in (folder/'summary.json', folder/'metrics.json', folder/'per_class.csv') if p.is_file()]
            accepted = False
            for attempt in range(3):
                accepted = notify_experiment(args.version, summary['status'], str(folder), summary['seconds'],
                                              summary=summary, attachments=attachments)
                if accepted:
                    break
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
            write(folder/'notification.json', dict(smtp_accepted=accepted, attempts=attempt+1,
                  attachments=[p.name for p in attachments], inbox_delivery_confirmed=False))
        lock.unlink()
        signal.signal(signal.SIGTERM, old_signal)
    if summary['status'] not in ('complete', 'ready'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
