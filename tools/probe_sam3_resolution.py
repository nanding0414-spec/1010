"""Isolated GPU probes for fixed masks, original/SPAN images and denser tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_probe_regions(image, mask):
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(image) as im:
        original = np.array(im.convert('RGB'))
    with Image.open(mask) as im:
        labels = np.array(im)
    if labels.shape != original.shape[:2] or not np.isin(labels, list(range(21))+[255]).all():
        raise ValueError(f'Class-ID mask must match original image dimensions: {mask}')
    regions, classes = [], []
    for cid in range(1,21):
        region_mask = (labels==cid).astype(np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA] >= 300:
                regions.append((components==i).astype(np.float32)); classes.append(cid)
    return original, regions, classes


def select_probe_sample(root, image=None, mask=None, span_index=None):
    from PIL import Image
    automatic = image is None
    if automatic and mask is not None:
        raise ValueError('--mask requires an explicit --image')
    split = root/'data/VOC2012/ImageSets/Segmentation/trainaug.txt'
    candidates = ([root/f'data/VOC2012/JPEGImages/{stem}.jpg' for stem in split.read_text().split()]
                  if automatic else [image])
    skipped = []
    for candidate in candidates:
        label = mask or root/'storage/pseudo_labels/voc/sam3_active'/f'{candidate.stem}.png'
        for path in (candidate, label):
            if not path.is_file():
                raise FileNotFoundError(path)
        span = None if span_index is None else span_index.get(candidate.stem)
        if span_index is not None and span is None:
            raise FileNotFoundError(f'Missing matching SPAN image: {candidate.stem}')
        original, regions, classes = load_probe_regions(candidate, label)
        if span is not None:
            with Image.open(span) as im:
                h, w = original.shape[:2]
                if im.height*w != im.width*h:
                    raise ValueError(f'SPAN aspect ratio differs: {span}')
        if regions:
            record = dict(policy='first_valid_trainaug' if automatic else 'explicit', image=str(candidate.resolve()),
                          mask=str(label.resolve()), region_count=len(regions), class_ids=classes, skipped=skipped)
            print(f'[Sample] {candidate.stem}: {len(regions)} regions; skipped_empty={len(skipped)}', flush=True)
            return candidate, label, span, record
        if not automatic:
            raise ValueError(f'Explicit sample {candidate.stem} has no surviving foreground; choose another --image/--mask')
        skipped.append(dict(image=str(candidate.resolve()), mask=str(label.resolve()), reason='no_surviving_foreground'))
        print(f'[Sample] skip {candidate.stem}: no foreground after area>=300', flush=True)
    raise ValueError('No valid foreground sample found in trainaug; no GPU workers started')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, help='Default: first trainaug image with surviving foreground; selection is logged')
    p.add_argument('--mask', type=Path, help='Class-ID PNG aligned to original image; default storage/pseudo_labels/voc/sam3_active')
    p.add_argument('--span-image-dir', type=Path, help='Existing SPAN x4 or x2 images, matched by stem; no SR is generated')
    p.add_argument('--resolutions', nargs='+', type=int, default=[1008, 1344, 2016])
    p.add_argument('--checkpoint', type=Path, default=ROOT/'weights/sam3/sam3.pt')
    p.add_argument('--output-dir', type=Path, default=ROOT/'experiments/sam3_resolution_probe_v1')
    p.add_argument('--cuda', default='1')
    p.add_argument('--query-chunk-size', type=int, default=256, help='High-resolution global attention query rows per chunk; 0=original SDPA')
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.query_chunk_size < 0:
        p.error('--query-chunk-size must be nonnegative')
    os.chdir(ROOT)
    if any(r < 336 or r % 336 for r in args.resolutions) or len(set(args.resolutions)) != len(args.resolutions):
        p.error('Use unique multiples of 336, e.g. 1008 1344 2016')
    from src.bank_input_validation import image_index
    args.image, args.mask, span, selection = select_probe_sample(
        ROOT, args.image, args.mask, image_index(args.span_image_dir) if args.span_image_dir else None)
    for path in [args.image, args.mask, args.checkpoint] + ([span] if span else []):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.worker is None:
        gpu = subprocess.check_output(['nvidia-smi', '--id='+args.cuda,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, gpu.strip().split(','))
        if memory > 1000 or utilization > 10:
            raise RuntimeError(f'GPU occupied: {gpu.strip()}. Wait for the current experiment to finish.')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        files = [args.image, args.mask, args.checkpoint, Path(__file__), ROOT/'src/sam3_resolution.py',
                 ROOT/'src/feature_extractor_sam3.py', ROOT/'src/feature_signature.py'] + ([span] if span else [])
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024), b''):
                    h.update(block)
            return h.hexdigest()
        write(args.output_dir/'manifest.json', dict(selection=selection, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            sha256={str(f.resolve()):digest(f) for f in files}, mask_policy='original coordinates, components >=300, no morphology',
            purpose='Mechanism/VRAM probe only; no MLP training or validation mIoU'))
        env = {k:v for k,v in os.environ.items() if not k.startswith('OVERRIDE_')}
        env['CUDA_VISIBLE_DEVICES'] = args.cuda
        statuses = []
        for resolution in args.resolutions:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(resolution),
                '--image', str(args.image.resolve()), '--mask', str(args.mask.resolve()),
                '--checkpoint', str(args.checkpoint.resolve()), '--output-dir', str(args.output_dir.resolve())]
            command += ['--query-chunk-size', str(args.query_chunk_size)]
            if args.span_image_dir:
                command += ['--span-image-dir', str(args.span_image_dir.resolve())]
            print(f'[Probe] {resolution}; log={args.output_dir / f"{resolution}.log"}', flush=True)
            with (args.output_dir/f'{resolution}.log').open('w') as stream:
                status = subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            statuses.append(dict(resolution=resolution, exit_code=status))
            print(f'[Probe] {resolution}: {"passed" if status == 0 else "FAILED; inspect log"}', flush=True)
            write(args.output_dir/'status.json', statuses)
        if any(s['exit_code'] for s in statuses):
            raise SystemExit(1)
        return
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    original, regions, classes = load_probe_regions(args.image, args.mask)
    if not regions:
        raise ValueError('No foreground regions survive the area filtering; choose another --image/--mask')
    torch.cuda.reset_peak_memory_stats()
    extractor = SAM3FeatureExtractor(model_path=str(args.checkpoint), device=torch.device('cuda:0'),
        resolution=args.worker, feature_source='trunk_fpn', feature_level=-1,
        token_norm='layernorm', context_subtract=0, use_bf16=True,
        enable_segmentation=False, use_interactivity=False, query_chunk_size=args.query_chunk_size)
    grid = (args.worker//14, args.worker//14)
    if extractor.feature_grid != grid:
        raise ValueError(f'Actual grid {extractor.feature_grid} != expected {grid}')
    weights = np.stack([cv2.resize(m, (grid[1],grid[0]), interpolation=cv2.INTER_AREA).reshape(-1) for m in regions])
    supported, mixed = weights>1e-6, (weights>1e-6)&(weights<1-1e-6)
    rows = [dict(class_id=cid, target_patches=int(s.sum()), mixed_patches=int(b.sum()),
                 mixed_fraction=float(b.sum()/max(s.sum(),1)), mixed_weight_fraction=float(w[b].sum()/max(w.sum(),1e-8)))
            for cid,s,b,w in zip(classes,supported,mixed,weights)]
    timings, features = {}, {}
    for name, path in [('original',args.image)] + ([('span',span)] if span else []):
        with Image.open(path) as im:
            image = np.array(im.convert('RGB'))
        if image.shape[0]*original.shape[1] != image.shape[1]*original.shape[0]:
            raise ValueError('SPAN image must preserve original aspect ratio')
        torch.cuda.synchronize(); start=time.monotonic()
        with torch.inference_mode():
            inp, _ = extractor._preprocess_image(image)
            tokens, actual_grid = extractor._forward_tokens(inp)
            if actual_grid != grid or not torch.isfinite(tokens).all():
                raise ValueError('Invalid feature grid or non-finite tokens')
            w = torch.as_tensor(weights, device=tokens.device, dtype=tokens.dtype)
            pooled = torch.nn.functional.normalize((w @ tokens[0]) / (w.sum(1, keepdim=True) + 1e-8), dim=1)
            if not torch.isfinite(pooled).all():
                raise ValueError('Non-finite region features')
            features[name] = pooled.cpu().numpy()
            del tokens, inp, pooled
        torch.cuda.synchronize()
        timings[name] = dict(seconds=time.monotonic()-start, image_size=list(image.shape[:2]))
    np.savez(args.output_dir/f'{args.worker}_features.npz', class_ids=np.array(classes), **features)
    write(args.output_dir/f'{args.worker}.json', dict(signature=extractor.feature_signature(), grid=list(grid),
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, regions=rows, timings=timings,
        note='Fixed original-coordinate regions; geometric statistics shared by original/SPAN at each grid; no accuracy claim'))


if __name__ == '__main__':
    main()
