#!/usr/bin/env python3
"""Standalone SAM3 proposal experiment: exactly three images per run.

Uses the project's proposer classes, without importing the feature/retrieval
pipeline or global configuration. Run --help for tuning options.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time


def parse_args(argv=None, *, bulk=False):
    p = argparse.ArgumentParser(description='Export SAM3 proposals in ModuSeg format.' if bulk else __doc__)
    p.add_argument('--image-dir', type=Path, default=Path('data/VOC2012/JPEGImages'))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--image-ids', nargs='+' if bulk else 3, help='Image stems' if bulk else 'Exactly three image stems')
    selection.add_argument('--split', type=Path, help='Image stems, one per line')
    p.add_argument('--offset', type=int, default=0, help='Skip this many images')
    if bulk:
        p.add_argument('--limit', type=int, default=0, help='0 exports all selected images')
    p.add_argument('--checkpoint', type=Path, default=Path('weights/sam3/sam3.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/sam3_proposals_active' if bulk else 'outputs/sam3_proposal_demo'))
    p.add_argument('--backend', choices=['grid', 'query', 'active'], default='active' if bulk else 'grid')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompts', default='object', help='Query backend only: prompts separated by |')
    p.add_argument('--confidence-threshold', type=float, default=0.15)
    p.add_argument('--grid-size', type=int, default=48)
    p.add_argument('--points-per-batch', type=int, default=64)
    p.add_argument('--pred-iou-threshold', type=float, default=0.3)
    p.add_argument('--stability-threshold', type=float, default=0.5)
    p.add_argument('--crop-n-layers', type=int, default=1)
    p.add_argument('--crop-points-downscale', type=int, default=2)
    p.add_argument('--filter-crop-edge', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--nms-threshold', type=float, default=0.7, help='Mask IoU NMS, also across crops')
    p.add_argument('--min-mask-area', type=int, default=100)
    p.add_argument('--min-mask-region-area', type=int, default=0)
    p.add_argument('--max-mask-area-ratio', type=float, default=None,
                   help='Default: grid=0.9, query=0.95; 0 disables upper area limit')
    p.add_argument('--max-masks', type=int, default=0, help='0 keeps all proposals')
    p.add_argument('--individual-top-k', type=int, default=24,
                   help='Individual previews only; 0 previews all. NPZ always contains all masks')
    active = p.add_argument_group('active backend only (experimental, foreground vocabulary required)')
    active.add_argument('--classes-file', type=Path,
                        default=Path(__file__).resolve().parent / 'tools/proposal_demo/voc_classes.json')
    active.add_argument('--bucket-size', type=int, default=4, help='Prompts per decoder batch; lower if GPU OOM')
    active.add_argument('--preview', action=argparse.BooleanOptionalAction, default=True)
    active.add_argument('--preview-resolution', type=int, default=672)
    active.add_argument('--presence-threshold', type=float, default=0.2)
    active.add_argument('--active-keep-ratio', type=float, default=0.4,
                        help='Keep at least this fraction, combined with presence threshold hits')
    active.add_argument('--min-active-classes', type=int, default=3)
    active.add_argument('--max-active-classes', type=int, default=12, help='0 disables cap')
    active.add_argument('--neighbor-tokens', type=int, default=2)
    active.add_argument('--hypernym-tokens', type=int, default=2)
    active.add_argument('--context-strength', type=float, default=1.0, help='0 disables extra context tokens')
    args = p.parse_args(argv)
    if bulk and args.limit < 0:
        p.error('limit must be >= 0')
    if args.max_mask_area_ratio is None:
        args.max_mask_area_ratio = 0.9 if args.backend == 'grid' else 0.95
    for name in ('confidence_threshold', 'pred_iou_threshold', 'stability_threshold',
                 'nms_threshold', 'max_mask_area_ratio', 'presence_threshold',
                 'active_keep_ratio'):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f'{name} must be in [0, 1]')
    for name in ('offset', 'crop_n_layers', 'min_mask_area', 'min_mask_region_area',
                 'max_masks', 'individual_top_k', 'max_active_classes', 'neighbor_tokens', 'hypernym_tokens'):
        if getattr(args, name) < 0:
            p.error(f'{name} must be >= 0')
    for name in ('grid_size', 'points_per_batch', 'crop_points_downscale', 'bucket_size', 'min_active_classes'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be >= 1')
    if not any(s.strip() for s in args.prompts.split('|')):
        p.error('prompts must not be empty')
    if args.image_ids and args.offset:
        p.error('--offset cannot be combined with --image-ids')
    if args.max_active_classes and args.max_active_classes < args.min_active_classes:
        p.error('max-active-classes must be 0 or >= min-active-classes')
    if args.preview_resolution % 28 or not 224 <= args.preview_resolution <= 1008:
        p.error('preview-resolution must be a multiple of 28 between 224 and 1008')
    if not math.isfinite(args.context_strength) or args.context_strength < 0:
        p.error('context-strength must be finite and >= 0')
    if args.backend == 'active' and args.prompts != 'object':
        p.error('active mode uses --classes-file; --prompts is for query mode')
    return args


def select_images(args):
    from tools.visualize_proposals import image_index

    images = image_index(args.image_dir)
    if args.image_ids:
        ids = args.image_ids
    else:
        ids = (args.split.read_text(encoding='utf-8-sig').splitlines()
               if args.split else list(images))
        ids = list(dict.fromkeys(i.strip() for i in ids if i.strip()))
        ids = ids[args.offset:args.offset + 3]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError('Each run requires three distinct images; check selection/offset')
    for image_id in ids:
        if image_id not in images:
            raise FileNotFoundError(f'Image ID not found: {image_id}')
    return [images[i] for i in ids]


def build_proposer(args, device):
    from sam3.model_builder import build_sam3_image_model

    classes = None
    if args.backend == 'active':
        from tools.proposal_demo.active_utils import load_vocabulary
        classes = load_vocabulary(args.classes_file)
    query = args.backend in ('query', 'active')
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint.resolve()), load_from_HF=False,
        device=device.type, eval_mode=True, enable_segmentation=query,
        enable_inst_interactivity=not query,
    ).to(device).eval()
    model.requires_grad_(False)
    if args.precision == 'fp32':
        from src.sam3_precision import use_standard_vit_mlp
        if not use_standard_vit_mlp(model):
            raise RuntimeError('No supported SAM3 ViT MLP found for FP32 proposals')
        model.float()
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    common = dict(model=model, device=device, min_mask_area=args.min_mask_area,
                  min_mask_region_area=args.min_mask_region_area,
                  max_mask_area_ratio=args.max_mask_area_ratio, max_masks=args.max_masks)
    if args.backend == 'active':
        from tools.proposal_demo.active_proposer import SAM3ActiveProposer
        import torch
        active_names = ('bucket_size', 'preview', 'preview_resolution', 'presence_threshold',
                        'active_keep_ratio', 'min_active_classes', 'max_active_classes',
                        'neighbor_tokens', 'hypernym_tokens', 'context_strength')
        amp = (torch.autocast('cuda', dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16)
               if device.type == 'cuda' and args.precision != 'fp32' else nullcontext())
        with amp:
            return SAM3ActiveProposer(**common, classes=classes,
                                      confidence_threshold=args.confidence_threshold,
                                      mask_nms_threshold=args.nms_threshold,
                                      **{name: getattr(args, name) for name in active_names})
    if query:
        from src.mask_proposer_sam3_query import SAM3QueryProposer
        return SAM3QueryProposer(
            **common, prompts=args.prompts, confidence_threshold=args.confidence_threshold,
            mask_nms_threshold=args.nms_threshold)
    from src.mask_proposer_sam3 import SAM3GridProposer
    return SAM3GridProposer(
        **common, grid_size=args.grid_size, points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_threshold, stability_score_thresh=args.stability_threshold,
        crop_n_layers=args.crop_n_layers, crop_n_points_downscale_factor=args.crop_points_downscale,
        filter_crop_edge=args.filter_crop_edge, box_nms_thresh=args.nms_threshold,
        crop_nms_thresh=args.nms_threshold)


def save_result(folder, rgb, masks, scores, individual_top_k):
    import numpy as np
    from PIL import Image
    from tools.visualize_proposals import render_overlay, select_proposals

    folder.mkdir()
    np.savez_compressed(folder / 'proposals.npz', masks=masks.astype(np.uint8), scores=scores)
    indices = select_proposals(masks, scores)
    overlay, occupied = render_overlay(rgb, masks, scores, indices, draw_labels=True)
    Image.fromarray(rgb).save(folder / 'original.png')
    Image.fromarray(overlay).save(folder / 'overlay.png')
    Image.fromarray(np.concatenate((rgb, overlay), axis=1)).save(folder / 'comparison.png')
    individual = folder / 'individual'
    individual.mkdir()
    for index in (indices[:individual_top_k] if individual_top_k else indices):
        preview, _ = render_overlay(rgb, masks, scores, [index], draw_labels=True)
        Image.fromarray(preview).save(individual / f'{index + 1:04d}_score_{scores[index]:.4f}.png')
    return dict(count=len(masks), coverage_ratio=float(occupied.mean()),
                scores=scores.tolist(), areas=masks.sum(axis=(1, 2)).tolist())


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    from PIL import Image, ImageOps

    images = select_images(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'SAM3 checkpoint not found: {args.checkpoint}')
    # Decode all three before loading the expensive model.
    rgbs = []
    for path in images:
        with Image.open(path) as source:
            rgbs.append(np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy())
    import torch

    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('SAM3 demo supports cpu or cuda devices')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use your SAM3 CUDA environment or --device cpu --precision fp32')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable on this GPU; choose --precision fp16 or fp32')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    folder = args.output_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.backend)
    folder.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['resolved_device'] = str(device)
    config['images'] = [str(p.resolve()) for p in images]
    (folder / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[Output] {folder.resolve()}', flush=True)
    proposer = build_proposer(args, device)
    summary = []
    for number, (path, rgb) in enumerate(zip(images, rgbs), 1):
        start = time.perf_counter()
        amp = (torch.autocast('cuda', dtype=getattr(torch, 'float16' if args.precision == 'fp16' else 'bfloat16'))
               if device.type == 'cuda' and args.precision != 'fp32' else nullcontext())
        with torch.inference_mode(), amp:
            instances = proposer(np.ascontiguousarray(rgb[..., ::-1]))['instances']
        masks = instances.pred_masks.detach().cpu().numpy().astype(bool)
        scores = instances.scores.detach().float().cpu().numpy()
        if not len(scores):
            masks = np.zeros((0, *rgb.shape[:2]), dtype=bool)
        elapsed = time.perf_counter() - start
        row = save_result(folder / path.stem, rgb, masks, scores, args.individual_top_k)
        if args.backend == 'active':
            from tools.proposal_demo.visualize import save_active_result
            save_active_result(folder / path.stem, rgb, masks, scores, proposer.last_result)
            row['active'] = proposer.last_result['report']
            names = ', '.join(row['active']['active_class_names'])
            print(f'  [Active] classes: {names}; proposals={len(scores)}', flush=True)
        row.update(image_id=path.stem, inference_seconds=elapsed)
        summary.append(row)
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'[{number}/3] {path.stem}: {len(scores)} proposals, {elapsed:.1f}s', flush=True)
    print(f'[Done] {folder.resolve()}')
    print('Colors identify proposals, not classes; coverage is not an accuracy metric.')


if __name__ == '__main__':
    main()
