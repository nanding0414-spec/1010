#!/usr/bin/env python3
"""Export all selected images as flat ModuSeg masks/scores NPZ caches."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
from zipfile import BadZipFile

from run_sam3_proposal_demo import build_proposer, parse_args


def select_images(args):
    from tools.visualize_proposals import image_index
    images = image_index(args.image_dir)
    ids = args.image_ids
    if ids is None:
        ids = (args.split.read_text(encoding='utf-8-sig').splitlines()
               if args.split else list(images))
    ids = list(dict.fromkeys(i.strip() for i in ids if i.strip()))[args.offset:]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError('No images selected')
    missing = [i for i in ids if i not in images]
    if missing:
        raise FileNotFoundError(f'Image IDs not found: {missing[:10]}')
    return [images[i] for i in ids]


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def file_identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare_output(args):
    ignored = {'output_dir', 'image_ids', 'split', 'offset', 'limit', 'individual_top_k', 'device'}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(args).items() if k not in ignored}
    config['checkpoint_identity'] = file_identity(args.checkpoint)
    if args.backend == 'active':
        config['vocabulary'] = json.loads(args.classes_file.read_text(encoding='utf-8-sig'))
    config['format_version'] = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'export_config.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        previous.pop('device', None)  # Accept manifests written before device-independent resume.
        if previous != config:
            raise ValueError('Export settings changed. Choose a new --output-dir to avoid mixing proposals.')
    else:
        if any(args.output_dir.glob('*.npz')):
            raise ValueError('Existing NPZ files have no export manifest. Choose a new --output-dir.')
        atomic_json(manifest, config)
    (args.output_dir / 'meta').mkdir(exist_ok=True)


def is_complete(folder, path):
    import numpy as np
    from PIL import Image
    receipt = folder / 'meta' / f'{path.stem}.json'
    cache = folder / f'{path.stem}.npz'
    if not receipt.is_file() or not cache.is_file():
        return False
    try:
        if json.loads(receipt.read_text(encoding='utf-8'))['source'] != file_identity(path):
            return False
        with Image.open(path) as img:
            width, height = img.size
        with np.load(cache, allow_pickle=False) as data:
            masks, scores = data['masks'], data['scores']
            return (masks.ndim == 3 and masks.shape[1:] == (height, width)
                    and scores.shape == (len(masks),) and np.isfinite(scores).all()
                    and np.all((masks == 0) | (masks == 1)))
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def main(argv=None):
    args = parse_args(argv, bulk=True)
    images = select_images(args)
    prepare_output(args)
    pending = [path for path in images if not is_complete(args.output_dir, path)]
    print(f'Selected: {len(images)} | complete: {len(images) - len(pending)} | pending: {len(pending)}', flush=True)
    print(f'Output: {args.output_dir.resolve()}', flush=True)
    if not pending:
        return 0

    import numpy as np
    import torch
    from PIL import Image
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda devices')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        count = torch.cuda.device_count()
        if device.index >= count:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
            raise ValueError(
                f'Requested {device}, but only {count} CUDA devices are visible. '
                f'CUDA_VISIBLE_DEVICES={visible!r}. Device indices are process-local; '
                'when CUDA_VISIBLE_DEVICES selects one GPU, use --device cuda:0.')
        torch.cuda.set_device(device)
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable; use --precision fp16 or fp32')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    proposer = build_proposer(args, device)
    for number, path in enumerate(pending, 1):
        # Per-image seed keeps resumed runs consistent with uninterrupted runs.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        receipt = args.output_dir / 'meta' / f'{path.stem}.json'
        receipt.unlink(missing_ok=True)
        source_identity = file_identity(path)
        with Image.open(path) as source:
            # Match ModuSeg's raw image dimensions (no EXIF rotation).
            rgb = np.asarray(source.convert('RGB')).copy()
        start = time.perf_counter()
        amp = (torch.autocast('cuda', dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16)
               if device.type == 'cuda' and args.precision != 'fp32' else nullcontext())
        with torch.inference_mode(), amp:
            instances = proposer(np.ascontiguousarray(rgb[..., ::-1]))['instances']
        masks = instances.pred_masks.detach().cpu().numpy().astype(np.uint8)
        scores = instances.scores.detach().float().cpu().numpy()
        if not len(scores):
            masks = np.zeros((0, *rgb.shape[:2]), dtype=np.uint8)
        if (masks.shape != (len(scores), *rgb.shape[:2]) or not np.isfinite(scores).all()
                or not np.all((masks == 0) | (masks == 1))):
            raise ValueError(f'Invalid proposals for {path.stem}')
        cache = args.output_dir / f'{path.stem}.npz'
        temporary = cache.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, masks=masks, scores=scores.astype(np.float32))
        temporary.replace(cache)
        report = dict(source=source_identity, count=len(scores), seconds=time.perf_counter() - start)
        if args.backend == 'active':
            report['active'] = proposer.last_result['report']
        atomic_json(receipt, report)
        print(f'[{number}/{len(pending)}] {path.stem}: {len(scores)} proposals, {report["seconds"]:.1f}s', flush=True)
    print(f'Done: {len(images)} images in {args.output_dir.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
