#!/usr/bin/env python3
"""Inference: SAM3 masks + SAM3/MLP features + feature-bank retrieval."""

import argparse
import inspect
import json
import os
from pathlib import Path
import time
from dataclasses import asdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import configs.config as cfg_module
from configs.config import (
    DATASET_CLASSES, DATASET_TYPE,
    INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT,
    INF_IGNORE_INDEX, INF_LOAD_PROPOSALS, INF_NMS_IOU, INF_NUM_CLASSES,
    INF_PROPOSALS_DIR, INF_SAVE_DIR, INF_TOPK_NEIGH, INF_VIS_LIMIT,
    INF_ALLOW_LEGACY_FEATURE_BANK,
    INF_MASK_BACKEND,
    PSEUDO_MASK_TYPE, SEED,
    FEATURE_BACKBONE,
)
from configs.config_helpers import get_mask_quality_str, get_model_version_str
from project_utils.config_logger import record_config
from project_utils.metrics import evaluate_dataset_miou
from project_utils.seed_utils import set_global_seed
from project_utils.seg_vis import overlay_segmentation, render_legend_panel
from src.inference import (
    build_faiss_index,
    init_mask_proposer,
    load_class_features,
    load_class_mapping,
    load_eval_samples,
    load_proposals,
    predict_instances,
    rasterize_instances,
    reduce_ade20k_gt_for_eval,
    reduce_ade20k_pred_for_eval,
    reduce_cityscapes_gt_for_eval,
    reduce_cityscapes_pred_for_eval,
)
from src.feature_signature import validate_feature_signature
from src.retrieval import build_retrieval_context
from src.feature_extractor_factory import build_feature_extractor
from src.sam3_runtime import build_sam3_model
from src.proposal_cleanup import CleanupConfig, clean_proposals


def load_strict_proposals(directory, image_id, height, width):
    path = Path(directory) / f"{image_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        masks, scores = data["masks"], data["scores"]
    if (masks.ndim != 3 or scores.shape != (len(masks),)
            or not np.isfinite(scores).all() or not np.all((masks == 0) | (masks == 1))):
        raise ValueError(f"Invalid proposal arrays: {path}")
    if masks.shape[1:] != (height, width) and not (len(masks) == 0 and masks.shape[1:] == (1, 1)):
        raise ValueError(f"Proposal/image dimensions differ: {path}")
    return [mask.astype(bool) for mask in masks], scores.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-config", type=Path, help="CleanupConfig JSON applied to cached proposals in memory")
    parser.add_argument("--results-json", type=Path, help="Machine-readable metrics, written only after successful evaluation")
    parser.add_argument("--output-dir", type=Path, help="Override evaluation/log/visualization directory")
    parser.add_argument("--strict-eval", action="store_true", help="Fail on missing inputs or feature-extraction errors")
    parser.add_argument("--eval-limit", type=int, default=0, help="First N validation images for diagnostics; 0=all")
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--feature-image-dir', type=Path, help='Alternate same-aspect images for feature extraction only')
    parser.add_argument('--feature-image-scale', type=int, choices=[1,4], default=4)
    parser.add_argument('--rescue', action='store_true', help='Enable HCR high-confidence retrieval rescue from the raw companion of a clean cache')
    parser.add_argument('--rescue-raw-dir', type=Path, help='Override the raw companion directory recorded in cleanup_config.json')
    args = parser.parse_args(argv)
    if args.rescue and (not args.strict_eval or not INF_LOAD_PROPOSALS or args.cleanup_config or DATASET_TYPE != 'voc'):
        parser.error('--rescue requires VOC, --strict-eval, cached clean proposals, and no --cleanup-config')
    if args.rescue_raw_dir and not args.rescue:
        parser.error('--rescue-raw-dir requires --rescue')
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid evaluation shard')
    if args.feature_image_dir and (DATASET_TYPE != 'voc' or not INF_LOAD_PROPOSALS):
        parser.error('Alternate feature images require VOC cached proposals')
    if args.eval_limit < 0:
        parser.error("--eval-limit must be nonnegative")
    if args.cleanup_config and not INF_LOAD_PROPOSALS:
        parser.error("--cleanup-config requires OVERRIDE_INF_LOAD_PROPOSALS=true")
    cleanup = (CleanupConfig(**json.loads(args.cleanup_config.read_text(encoding="utf-8")))
               if args.cleanup_config else None)
    if args.results_json and args.results_json.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {args.results_json}")
    print("=" * 80)
    print(f"Inference - {DATASET_TYPE.upper()} | feature model: {FEATURE_BACKBONE}")
    print("=" * 80)

    set_global_seed(SEED)

    dataset_root = INF_DATASET_ROOT
    feature_bank_root = INF_FEATURE_BANK_ROOT
    sample_infos = load_eval_samples(dataset_root)
    if args.eval_limit:
        sample_infos = sample_infos[:args.eval_limit]
    sample_infos = sample_infos[args.shard_index::args.shard_count]
    rescue_source = rescue_config = None
    if args.rescue:
        from src.semantic_rescue import HCRConfig, RescueSource
        rescue_config = HCRConfig()
        rescue_source = RescueSource(INF_PROPOSALS_DIR, [s['id'] for s in sample_infos], args.rescue_raw_dir)
        print(f'[HCR] thresholds={asdict(rescue_config)}; raw={rescue_source.raw_dir}; clean masks unchanged')
    feature_images = None
    if args.feature_image_dir:
        from src.bank_input_validation import image_index
        feature_images = image_index(args.feature_image_dir)
        for sample in sample_infos:
            if sample['id'] not in feature_images:
                raise FileNotFoundError(f"Missing feature image: {sample['id']}")
    if args.strict_eval:
        if not sample_infos or len({sample['id'] for sample in sample_infos}) != len(sample_infos):
            raise ValueError("Evaluation list is empty or contains duplicate images")
        for sample in sample_infos:
            for key in ("img", "gt"):
                if not sample.get(key) or not Path(sample[key]).is_file():
                    raise FileNotFoundError(f"Missing {key} for validation image {sample['id']}")
            if INF_LOAD_PROPOSALS:
                proposal = Path(INF_PROPOSALS_DIR) / f"{sample['id']}.npz"
                if not proposal.is_file():
                    raise FileNotFoundError(proposal)

    # Load feature bank
    _, fg_names, bg_names = load_class_mapping(os.path.join(feature_bank_root, 'meta'))
    class_to_feats = load_class_features(
        os.path.join(feature_bank_root, 'features'),
        list(dict.fromkeys(fg_names + bg_names)),
    )
    assert len(class_to_feats) > 0, "Empty feature bank. Please run run_build_feature_bank.py first."

    # Build gallery features and FAISS index
    gallery_feats = []
    gallery_labels = []
    gallery_dims = {}
    for name, mat in class_to_feats.items():
        if mat is None or mat.shape[0] == 0:
            continue
        if mat.ndim != 2:
            raise RuntimeError(
                f"Feature matrix for {name!r} must be 2-D, got shape {mat.shape}"
            )
        gallery_dims[name] = int(mat.shape[1])
        mat = mat.astype(np.float32)
        valid = np.all(np.isfinite(mat), axis=1) & (np.linalg.norm(mat, axis=1) >= 1e-6)
        if not np.all(valid):
            print(f"[FeatureBank] Dropped {int((~valid).sum())} invalid vectors from {name}")
        mat = mat[valid]
        if mat.shape[0] == 0:
            continue
        gallery_feats.append(mat)
        gallery_labels.extend([name] * mat.shape[0])
    if len(gallery_feats) == 0:
        raise RuntimeError("No gallery features available.")
    unique_dims = sorted(set(gallery_dims.values()))
    if len(unique_dims) != 1:
        details = ", ".join(f"{name}={dim}" for name, dim in gallery_dims.items())
        raise RuntimeError(
            f"Feature bank contains mixed dimensions ({details}). Rebuild it in a clean output directory."
        )

    all_feats = np.vstack(gallery_feats)
    all_feats = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    all_feats = np.ascontiguousarray(all_feats.astype(np.float32))

    # Cached proposals skip only the mask proposer; SAM3 features are still needed.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = build_feature_extractor(device, use_interactivity=False)
    validate_feature_signature(
        os.path.join(feature_bank_root, 'meta'),
        dino.feature_signature(),
        allow_missing=INF_ALLOW_LEGACY_FEATURE_BANK,
    )
    if all_feats.shape[1] != dino.embed_dim:
        raise RuntimeError(
            f"Feature bank dim {all_feats.shape[1]} != extractor dim {dino.embed_dim}. "
            "Rebuild the feature bank."
        )
    retrieval_context = build_retrieval_context(all_feats, gallery_labels)
    faiss_index = build_faiss_index(all_feats)
    model_version = get_model_version_str()

    if INF_LOAD_PROPOSALS:
        proposals_dir = os.path.abspath(INF_PROPOSALS_DIR)
        assert os.path.isdir(proposals_dir), (
            f"Precomputed proposals dir not found: {proposals_dir}. "
            f"Run run_generate_proposals.py first, or set OVERRIDE_INF_LOAD_PROPOSALS=false."
        )
        mask_proposer = None
        print(f"[Inference] LOAD_PROPOSALS=true: loading from {proposals_dir} (SAM3 proposer skipped)")
    else:
        proposals_dir = None
        use_query_backend = INF_MASK_BACKEND == "sam3_query"
        sam3_model = build_sam3_model(
            device=device,
            enable_segmentation=use_query_backend,
            enable_inst_interactivity=not use_query_backend,
        )
        mask_proposer = init_mask_proposer(sam3_model=sam3_model)
    save_dir = str(args.output_dir) if args.output_dir else os.path.join(INF_FEATURE_BANK_ROOT, INF_SAVE_DIR)
    record_config(save_dir if args.output_dir else os.path.join(INF_FEATURE_BANK_ROOT, 'meta'), 'run_inference.py')
    os.makedirs(save_dir, exist_ok=True)

    preds = []
    gts = []
    evaluated_ids = []
    proposal_count_before = proposal_count_after = 0
    rescue_reports = []

    for sample in tqdm(sample_infos, desc=f"Infer ({model_version})", ncols=100):
        img_id = sample['id']
        img_path = sample['img']
        gt_path = sample['gt']

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            if args.strict_eval:
                raise OSError(f"Cannot decode validation image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        feature_rgb = None
        if feature_images is not None:
            with Image.open(feature_images[img_id]) as feature_image:
                feature_rgb = np.array(feature_image.convert('RGB'))
            if feature_rgb.shape[:2] != (h * args.feature_image_scale, w * args.feature_image_scale):
                raise ValueError(f'Expected exact x{args.feature_image_scale} feature image for {img_id}')

        gt = None
        base_extractor = getattr(dino, 'base', dino)
        if getattr(base_extractor, 'image_token_cache', None) is not None:
            feature_input = feature_rgb if feature_rgb is not None else img_rgb
            base_extractor.set_image_source(feature_images[img_id] if feature_images is not None else img_path, feature_input)
            base_extractor.cache_image_features(feature_input)
        if gt_path and os.path.isfile(gt_path):
            gt = np.array(Image.open(gt_path))
            if args.strict_eval and gt.shape != (h, w):
                raise ValueError(f"GT/image dimensions differ: {gt_path}")

        if INF_LOAD_PROPOSALS:
            pre_masks, pre_scores = (load_strict_proposals(proposals_dir, img_id, h, w)
                                     if args.strict_eval else load_proposals(proposals_dir, img_id))
            if pre_masks is None:
                print(f"\n[WARN] no precomputed proposals for {img_id}, skipping")
                continue
            proposal_count_before += len(pre_masks)
            if cleanup is not None:
                mask_array = np.stack(pre_masks) if pre_masks else np.zeros((0, h, w), dtype=bool)
                cleaned, scores, _, _ = clean_proposals(mask_array, pre_scores, cleanup)
                pre_masks, pre_scores = list(cleaned), scores.tolist()
            proposal_count_after += len(pre_masks)
            rescue_kwargs = {}
            if rescue_source is not None:
                pre_masks, pre_scores, flags, rescue_record = rescue_source.expand(img_id, pre_masks, pre_scores, h, w)
                rescue_stats = {}
                rescue_kwargs = dict(rescue_flags=flags, rescue_config=rescue_config, rescue_stats=rescue_stats)
            instances = predict_instances(
                img_rgb,
                dino,
                mask_proposer=None,
                all_feats=all_feats,
                all_labels=gallery_labels,
                faiss_index=faiss_index,
                nms_iou=INF_NMS_IOU,
                topk_neigh=INF_TOPK_NEIGH,
                precomputed_masks=pre_masks,
                feature_image_rgb=feature_rgb,
                precomputed_scores=pre_scores,
                retrieval_context=retrieval_context,
                strict_errors=args.strict_eval,
                **rescue_kwargs,
            )
            if rescue_source is not None:
                rescue_reports.append(dict(rescue_record, selection=rescue_stats))
        else:
            instances = predict_instances(
                img_rgb,
                dino,
                mask_proposer,
                all_feats,
                gallery_labels,
                faiss_index,
                INF_NMS_IOU,
                topk_neigh=INF_TOPK_NEIGH,
                retrieval_context=retrieval_context,
                strict_errors=args.strict_eval,
            )

        pred_before = rasterize_instances(instances, h, w)

        if gt is not None:
            evaluated_ids.append(img_id)
            if DATASET_TYPE == "ade20k":
                preds.append(reduce_ade20k_pred_for_eval(pred_before))
                gts.append(reduce_ade20k_gt_for_eval(gt))
            elif DATASET_TYPE == "cityscapes":
                pred_eval = reduce_cityscapes_pred_for_eval(pred_before)
                gt_eval = reduce_cityscapes_gt_for_eval(gt)
                gt_eval[pred_eval == INF_IGNORE_INDEX] = INF_IGNORE_INDEX
                preds.append(pred_eval)
                gts.append(gt_eval)
            else:
                preds.append(pred_before)
                gts.append(gt)

        # Save visualization for the first N evaluated images.
        vis_count = len([p for p, g in zip(preds, gts) if g is not None])
        if vis_count <= INF_VIS_LIMIT:
            overlay = overlay_segmentation(img_rgb, pred_before)
            present_ids = sorted(list(set(int(v) for v in np.unique(pred_before) if v not in (0, 255))))
            panel = render_legend_panel(present_ids, overlay.shape[0])
            canvas = np.full((overlay.shape[0], overlay.shape[1] + panel.shape[1], 3), 255, dtype=np.uint8)
            canvas[:, :overlay.shape[1]] = overlay
            canvas[:, overlay.shape[1]:overlay.shape[1] + panel.shape[1]] = panel
            save_path = os.path.join(save_dir, f'{img_id}.jpg')
            cv2.imwrite(save_path, canvas)

    # Evaluate
    if args.strict_eval and len(preds) != len(sample_infos):
        raise RuntimeError("Incomplete evaluation; refusing to report comparable metrics")
    if len(preds) == 0:
        print(f"\nNo samples for evaluation. Results saved to: {save_dir}")
        return

    print(f"\n[Eval] {len(preds)} samples ...")
    eval_num_classes = len(DATASET_CLASSES) if DATASET_TYPE in ("ade20k", "cityscapes") else INF_NUM_CLASSES
    eval_class_names = DATASET_CLASSES if DATASET_TYPE in ("ade20k", "cityscapes") else None
    summary, metrics, class_names = evaluate_dataset_miou(
        preds,
        gts,
        eval_num_classes,
        INF_IGNORE_INDEX,
        class_names=eval_class_names,
    )
    print(summary)

    per_class = metrics['per_class_iou']
    print("\nPer-class IoU Table:")
    print("+----------------------+---------+")
    print("| Class                | IoU     |")
    print("+----------------------+---------+")
    for name, iou_val in zip(class_names, per_class):
        print(f"| {name:20s} | {iou_val:7.4f} |")
    print("+----------------------+---------+")
    print(f"| {'Mean(mIoU)':20s} | {metrics['mIoU']:7.4f} |")
    print("+----------------------+---------+")

    # Save results
    mask_quality = get_mask_quality_str()
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_txt = os.path.join(save_dir, f"{ts}_{DATASET_TYPE}_{mask_quality}_{model_version}_results.txt")

    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write(f"{DATASET_TYPE.upper()} - {mask_quality} Inference Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {DATASET_TYPE.upper()}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mask Quality: {mask_quality}\n")
        f.write(f"Feature Extractor: {model_version}\n")
        f.write(f"Feature Bank Root: {INF_FEATURE_BANK_ROOT}\n")
        f.write(f"Save Dir: {save_dir}\n")
        f.write(f"Pseudo Mask Type: {PSEUDO_MASK_TYPE}\n")
        f.write(f"Semantic Rescue (HCR): {'enabled' if rescue_source is not None else 'disabled'}\n")
        if rescue_source is not None:
            f.write(json.dumps(dict(rescue_source.description, parameters=asdict(rescue_config)), ensure_ascii=False) + '\n')
        f.write("-" * 40 + "\n")

        for name, value in vars(cfg_module).items():
            if name.startswith('__') or inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                continue
            f.write(f"{name}: {value}\n")

        f.write("-" * 40 + "\n")
        f.write(summary + "\n")
        f.write("\nPer-class IoU Table:\n")
        f.write("+----------------------+---------+\n")
        f.write("| Class                | IoU     |\n")
        f.write("+----------------------+---------+\n")
        for name, iou_val in zip(class_names, per_class):
            f.write(f"| {name:20s} | {iou_val:7.4f} |\n")
        f.write("+----------------------+---------+\n")
        f.write(f"| {'Mean(mIoU)':20s} | {metrics['mIoU']:7.4f} |\n")
        f.write("+----------------------+---------+\n")

    print(f"\nResults saved to {result_txt}")
    if args.results_json:
        confusion = np.zeros((eval_num_classes, eval_num_classes), dtype=np.int64)
        for pred, gt in zip(preds, gts):
            valid = (gt != INF_IGNORE_INDEX) & (gt >= 0) & (gt < eval_num_classes)
            confusion += np.bincount((gt[valid].astype(np.int64) * eval_num_classes + pred[valid]).reshape(-1),
                                     minlength=eval_num_classes**2).reshape(eval_num_classes, eval_num_classes)
        payload = {
            'confusion_matrix': confusion.tolist(), 'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'feature_image_dir': str(args.feature_image_dir.resolve()) if args.feature_image_dir else None,
            'feature_image_scale': args.feature_image_scale if args.feature_image_dir else 1,
            "dataset": DATASET_TYPE, "mIoU": float(metrics["mIoU"]),
            "per_class_iou": dict(zip(class_names, map(float, metrics["per_class_iou"]))),
            "evaluated_ids": evaluated_ids, "evaluated_count": len(evaluated_ids),
            "eval_limit": args.eval_limit, "cleanup": cleanup.to_dict() if cleanup else None,
            "proposal_count_before": proposal_count_before, "proposal_count_after": proposal_count_after,
            "feature_signature": dino.feature_signature(),
            "feature_bank": str(Path(INF_FEATURE_BANK_ROOT).resolve()),
            "proposals_dir": str(Path(proposals_dir).resolve()) if proposals_dir else None,
            "inference_settings": {
                name: value for name, value in vars(cfg_module).items()
                if (name.startswith("INF_") or name in {"MIN_MASK_AREA", "SEED"})
                and isinstance(value, (str, int, float, bool))
            },
        }
        if rescue_source is not None:
            payload['semantic_rescue'] = dict(rescue_source.description, parameters=asdict(rescue_config),
                                              images=rescue_reports,
                                              accepted=sum(x['selection'].get('accepted', 0) for x in rescue_reports),
                                              added_pixels=sum(x['selection'].get('added_pixels', 0) for x in rescue_reports),
                                              note='Proposal counts above describe the unchanged clean input; extra counts are per image here.')
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.results_json.with_name(args.results_json.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.results_json)


if __name__ == '__main__':
    main()
