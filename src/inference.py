"""Inference module: SAM3 proposals + selectable features + bank retrieval."""
import os
import json
from typing import List, Dict, Any
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.feature_bank_storage import load_feature_matrix_npz
from project_utils.seed_utils import set_global_seed
from configs.config import (
    INF_USE_FAISS, INF_FAISS_USE_GPU, INF_FAISS_DEVICE, INF_FAISS_EXACT_SEARCH_THRESHOLD,
    DATASET_TYPE,
    INF_MASK_BACKEND,
    INF_INSTANCE_SCORE_THR,
    INF_RETRIEVAL_MODE, INF_CLASS_TOPM, INF_RETRIEVAL_SEARCH_MULTIPLIER,
    INF_CENTROID_WEIGHT, INF_RETRIEVAL_MARGIN_THR, INF_PROPOSAL_SCORE_WEIGHT,
    MIN_MASK_AREA, DATASET_CLASSES, SEED,
    SAM3_GRID_SIZE, SAM3_POINTS_PER_BATCH, SAM3_PRED_IOU_THRESH,
    SAM3_STABILITY_THRESH, SAM3_MIN_MASK_REGION_AREA, SAM3_MAX_MASK_AREA_RATIO,
    SAM3_BOX_NMS_THRESH,
    SAM3_CROP_N_LAYERS, SAM3_CROP_OVERLAP_RATIO, SAM3_CROP_N_POINTS_DOWNSCALE,
    SAM3_CROP_NMS_THRESH, SAM3_FILTER_CROP_EDGE, SAM3_MAX_MASKS,
    SAM3_QUERY_PROMPTS, SAM3_QUERY_CONF_THRESH, SAM3_QUERY_NMS_THRESH,
    SAM3_QUERY_MAX_MASK_AREA_RATIO, SAM3_QUERY_MAX_MASKS, SAM3_RESOLUTION,
)
from src.retrieval import build_retrieval_context, classify_queries

# ============================================================================
# Mask proposer and feature bank loading
# ============================================================================

def init_mask_proposer(sam3_model=None):
    """Initialize the configured SAM3 proposal backend."""
    set_global_seed(SEED, verbose=False)

    backend = (INF_MASK_BACKEND or "sam3_grid").lower()
    if backend == "sam3_grid":
        from src.mask_proposer_sam3 import SAM3GridProposer
        assert sam3_model is not None, (
            "INF_MASK_BACKEND='sam3_grid' requires the SAM3 feature extractor model"
        )
        return SAM3GridProposer(
            model=sam3_model,
            grid_size=SAM3_GRID_SIZE,
            points_per_batch=SAM3_POINTS_PER_BATCH,
            pred_iou_thresh=SAM3_PRED_IOU_THRESH,
            stability_score_thresh=SAM3_STABILITY_THRESH,
            min_mask_region_area=SAM3_MIN_MASK_REGION_AREA,
            max_mask_area_ratio=SAM3_MAX_MASK_AREA_RATIO,
            box_nms_thresh=SAM3_BOX_NMS_THRESH,
            crop_n_layers=SAM3_CROP_N_LAYERS,
            crop_overlap_ratio=SAM3_CROP_OVERLAP_RATIO,
            crop_n_points_downscale_factor=SAM3_CROP_N_POINTS_DOWNSCALE,
            crop_nms_thresh=SAM3_CROP_NMS_THRESH,
            filter_crop_edge=SAM3_FILTER_CROP_EDGE,
            max_masks=SAM3_MAX_MASKS,
        )

    if backend == "sam3_query":
        from src.mask_proposer_sam3_query import SAM3QueryProposer
        assert sam3_model is not None, (
            "INF_MASK_BACKEND='sam3_query' requires a segmentation-enabled SAM3 model"
        )
        return SAM3QueryProposer(
            model=sam3_model,
            prompts=SAM3_QUERY_PROMPTS,
            confidence_threshold=SAM3_QUERY_CONF_THRESH,
            mask_nms_threshold=SAM3_QUERY_NMS_THRESH,
            min_mask_area=MIN_MASK_AREA,
            min_mask_region_area=SAM3_MIN_MASK_REGION_AREA,
            max_mask_area_ratio=SAM3_QUERY_MAX_MASK_AREA_RATIO,
            max_masks=SAM3_QUERY_MAX_MASKS,
            resolution=SAM3_RESOLUTION,
        )

    raise ValueError(
        f"Unknown INF_MASK_BACKEND: {INF_MASK_BACKEND}. "
        "Supported backends: sam3_grid, sam3_query."
    )


def load_class_mapping(meta_dir: str):
    """Load class mapping from meta directory."""
    path = os.path.join(meta_dir, 'class_mapping.json')
    assert os.path.isfile(path), f"Mapping not found: {path}. Please run run_build_feature_bank.py first."
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    id_to_dirname = {int(k): v for k, v in meta['id_to_dirname'].items()}
    fg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if i != 0 and not id_to_dirname[i].startswith('background')]
    bg_class_names = [id_to_dirname[i] for i in sorted(id_to_dirname.keys()) 
                      if id_to_dirname[i] == 'background' or id_to_dirname[i].startswith('background_')]
    return id_to_dirname, fg_class_names, bg_class_names


def load_class_features(features_dir: str, class_names: List[str]):
    """Load matrix-format class features."""
    out = {}
    for name in class_names:
        cls_dir = Path(features_dir) / name
        if not cls_dir.is_dir():
            continue
        
        npz_files = sorted(cls_dir.glob('*.npz'))
        if not npz_files:
            continue
        if len(npz_files) != 1:
            raise RuntimeError(f"Expected exactly one v2 feature file under {cls_dir}, got {len(npz_files)}")

        features, _sample_ids = load_feature_matrix_npz(npz_files[0])
        if features.shape[0] > 0:
            out[name] = features
    return out

# ============================================================================
# Core inference logic
# ============================================================================

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / (float(union) + 1e-8)


def nms_masks(candidates: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    """NMS filtering within a single class."""
    keep = []
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    for c in candidates:
        if c['mask'].sum() < MIN_MASK_AREA:
            continue
        ok = True
        for k in keep:
            if mask_iou(c['mask'], k['mask']) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def nms_masks_per_class(candidates: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    """Per-class NMS to avoid cross-class suppression."""
    from collections import defaultdict
    
    # Group by class
    cls_groups = defaultdict(list)
    for c in candidates:
        cls_groups[c['cls_name']].append(c)
    
    # Per-class NMS
    keep_all = []
    for cls_name, cls_candidates in cls_groups.items():
        keep = nms_masks(cls_candidates, iou_thr)
        keep_all.extend(keep)
    
    return keep_all


def generate_proposals(img_rgb: np.ndarray, mask_proposer):
    """Stage 1 only: SAM3 proposals -> score threshold -> area filter.

    Returns (masks: list[np.ndarray bool], scores: list[float]).
    Extracted from predict_instances so the proposal stage can be run and
    saved separately from the retrieval/evaluation stage.
    """
    img_bgr = img_rgb[..., ::-1].copy()
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        predictions = mask_proposer(img_bgr)

    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= INF_INSTANCE_SCORE_THR)
    selected_masks = pred_masks[selected_indexes]
    selected_scores = pred_scores[selected_indexes]

    masks_original = []
    scores_out = []
    for m, s in zip(selected_masks, selected_scores):
        m_np = m.detach().cpu().numpy().astype(np.uint8)
        if m_np.sum() < MIN_MASK_AREA:
            continue
        masks_original.append(m_np.astype(bool))
        scores_out.append(float(s.detach().cpu()))
    return masks_original, scores_out


def save_proposals(save_dir: str, img_id: str,
                   masks: List[np.ndarray], scores: List[float]):
    """Save per-image proposal masks + scores as a compressed npz."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{img_id}.npz")
    if masks:
        masks_arr = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    else:
        masks_arr = np.zeros((0, 1, 1), dtype=np.uint8)
    scores_arr = np.array(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32)
    np.savez_compressed(path, masks=masks_arr, scores=scores_arr)


def load_proposals(load_dir: str, img_id: str):
    """Load per-image proposal masks + scores from npz. Returns (masks, scores) or (None, None) if missing."""
    path = os.path.join(load_dir, f"{img_id}.npz")
    if not os.path.isfile(path):
        return None, None
    data = np.load(path)
    masks_arr = data['masks']
    scores_arr = data['scores']
    if masks_arr.shape[0] == 0:
        return [], []
    masks = [m.astype(bool) for m in masks_arr]
    scores = [float(s) for s in scores_arr]
    return masks, scores


def predict_instances(img_rgb: np.ndarray,
                      dino,  # SAM3FeatureExtractor
                      mask_proposer=None,
                      all_feats: np.ndarray = None,
                      all_labels: List[str] = None,
                      faiss_index=None,
                      nms_iou: float = 0.4,
                      topk_neigh: int = 9,
                      precomputed_masks: List[np.ndarray] = None,
                      precomputed_scores: List[float] = None,
                      retrieval_context=None,
                      strict_errors: bool = False,
                      feature_image_rgb=None,
                      rescue_flags=None,
                      rescue_config=None,
                      rescue_stats=None,
                      capture_clean_sources=False) -> List[Dict[str, Any]]:
    """Full inference: proposals -> feature extraction -> retrieval -> per-class NMS.

    If precomputed_masks is provided (list of bool masks), the SAM3 proposal
    stage is skipped and retrieval starts directly from feature extraction.
    """
    # 1) Generate candidate masks (or use precomputed)
    if precomputed_masks is not None:
        masks_original = list(precomputed_masks)
        if precomputed_scores is None:
            proposal_scores = [1.0] * len(masks_original)
        else:
            if len(precomputed_scores) != len(masks_original):
                raise ValueError("Precomputed proposal masks/scores length mismatch")
            proposal_scores = [float(x) for x in precomputed_scores]
    else:
        assert mask_proposer is not None, "mask_proposer required when precomputed_masks is None"
        masks_original, proposal_scores = generate_proposals(img_rgb, mask_proposer)

    if rescue_config is not None:
        if (precomputed_masks is None or rescue_flags is None
                or len(rescue_flags) != len(masks_original)
                or any(type(x) not in (bool, np.bool_) for x in rescue_flags)):
            raise ValueError('HCR requires aligned cached-proposal rescue flags')
    elif rescue_flags is not None:
        raise ValueError('Rescue flags require an HCR configuration')
    if len(masks_original) == 0 or all_feats is None or all_feats.shape[0] == 0:
        return []

    # Geometry, NMS and rasterization retain original-image coordinates.
    feature_rgb = img_rgb if feature_image_rgb is None else feature_image_rgb
    if feature_rgb.shape[0] * img_rgb.shape[1] != feature_rgb.shape[1] * img_rgb.shape[0]:
        raise ValueError('Feature image must preserve proposal-image aspect ratio')
    # 2) Batch feature extraction
    base_extractor = getattr(dino, "base", dino)
    failures_before = getattr(base_extractor, "_feature_failure_count", 0)
    feature_list = []
    if hasattr(dino, 'extract_features_batch'):
        try:
            feature_list = dino.extract_features_batch(feature_rgb, masks_original)
        except Exception as exc:
            if strict_errors:
                raise
            print(f"[Inference] batch feature extraction failed, retrying per mask: {exc}")
            feature_list = []
    
    if len(feature_list) != len(masks_original):
        feature_list = []
        for m in masks_original:
            try:
                feats = dino.extract_features(feature_rgb, m)
                vec = feats.detach().cpu().numpy().reshape(-1)
                feature_list.append(vec)
            except Exception as exc:
                if strict_errors:
                    raise
                if hasattr(dino, '_warn_feature_failure'):
                    dino._warn_feature_failure(exc)
                feature_list.append(np.zeros(1, dtype=np.float32))
    
    if strict_errors and getattr(base_extractor, "_feature_failure_count", 0) != failures_before:
        raise RuntimeError("Feature extraction reported errors during strict evaluation")

    # Stack and normalize features
    valid_features = []
    valid_masks_original = []
    valid_proposal_scores = []
    valid_rescue_flags = []
    for index, (m_orig, proposal_score, vec) in enumerate(zip(masks_original, proposal_scores, feature_list)):
        vec = np.asarray(vec).reshape(-1)
        if vec.shape[0] != all_feats.shape[1]:
            if strict_errors:
                raise ValueError("Feature dimension mismatch during strict evaluation")
            continue
        if strict_errors and not np.all(np.isfinite(vec)):
            raise ValueError("Non-finite feature during strict evaluation")
        if not np.all(np.isfinite(vec)) or np.linalg.norm(vec) < 1e-6:
            continue
        valid_features.append(vec)
        valid_masks_original.append(m_orig)
        valid_proposal_scores.append(float(proposal_score))
        valid_rescue_flags.append(bool(rescue_flags[index]) if rescue_config is not None else False)
    
    if len(valid_features) == 0:
        return []
    
    # Normalize query features
    query_feats = np.vstack([f.reshape(1, -1) for f in valid_features]).astype(np.float32)
    query_feats = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    query_feats = np.ascontiguousarray(query_feats)

    # 3) Class-balanced retrieval. The legacy vote remains available through
    # OVERRIDE_INF_RETRIEVAL_MODE=legacy for exact reproduction.
    if retrieval_context is None:
        retrieval_context = build_retrieval_context(all_feats, all_labels)
    retrieval_predictions = classify_queries(
        query_feats=query_feats,
        all_feats=all_feats,
        all_labels=all_labels,
        faiss_index=faiss_index,
        context=retrieval_context,
        topk=topk_neigh,
        mode=INF_RETRIEVAL_MODE,
        class_topm=INF_CLASS_TOPM,
        search_multiplier=INF_RETRIEVAL_SEARCH_MULTIPLIER,
        centroid_weight=INF_CENTROID_WEIGHT,
    )

    # 4) Combine semantic confidence with SAM3's predicted mask quality.
    candidates = []
    proposal_weight = float(np.clip(INF_PROPOSAL_SCORE_WEIGHT, 0.0, 1.0))
    extra_candidates = []
    for m, proposal_score, prediction, is_rescue in zip(
        valid_masks_original, valid_proposal_scores, retrieval_predictions, valid_rescue_flags
    ):
        if prediction.margin < globals().get('EXPERIMENT_CLASS_MARGINS', {}).get(prediction.label, INF_RETRIEVAL_MARGIN_THR):
            continue
        score = (
            (1.0 - proposal_weight) * prediction.score
            + proposal_weight * proposal_score
        )
        (extra_candidates if is_rescue else candidates).append({
            'mask': m,
            'cls_name': prediction.label,
            'score': float(score),
            'retrieval_score': float(prediction.score),
            'retrieval_margin': float(prediction.margin),
            'proposal_score': float(proposal_score),
        })

    # 5) Per-class NMS
    final_instances = nms_masks_per_class(candidates, nms_iou)
    clean_source_ids = {id(row) for row in final_instances} if capture_clean_sources else None
    if rescue_config is not None:
        from src.semantic_rescue import rescue_instances
        final_instances = rescue_instances(final_instances, extra_candidates, rescue_config,
                                           DATASET_CLASSES, rescue_stats)
    if capture_clean_sources:
        final_instances = [dict(row, geometric_completion_source=id(row) in clean_source_ids)
                           for row in final_instances]
    
    # 6) Sort by score descending
    final_instances = sorted(final_instances, key=lambda x: x['score'], reverse=True)
    
    return final_instances


def rasterize_instances(instances: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Convert instance list to semantic segmentation map."""
    label_map = {name: i+1 for i, name in enumerate(DATASET_CLASSES)}
    sem = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        cls = inst['cls_name']
        if cls == 'background' or cls not in label_map:
            continue
        m = inst['mask']
        sem[m & (sem == 0)] = label_map[cls]
    return sem


def reduce_ade20k_gt_for_eval(gt_seg: np.ndarray) -> np.ndarray:
    """Convert ADE20K GT from 0/1..150 to 255/0..149."""
    gt = np.asarray(gt_seg).astype(np.int64)
    out = np.full(gt.shape, 255, dtype=np.int64)
    valid = (gt >= 1) & (gt <= 150)
    out[valid] = gt[valid] - 1
    return out


def reduce_ade20k_pred_for_eval(pred_seg: np.ndarray) -> np.ndarray:
    """Convert internal ADE20K prediction labels from 0..150 to 0..149."""
    pred = np.asarray(pred_seg).astype(np.int64)
    return np.clip(pred - 1, 0, 149)


_CITYSCAPES_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    _CITYSCAPES_ID_TO_TRAIN_ID[_label_id] = _train_id


def cityscapes_image_to_gt_path(img_path: Path, dataset_root: Path) -> Path:
    """Infer the Cityscapes labelIds path from a leftImg8bit image path."""
    img_path = Path(img_path)
    split = img_path.parts[img_path.parts.index("leftImg8bit") + 1]
    city = img_path.parent.name
    base = img_path.stem.replace("_leftImg8bit", "")
    return dataset_root / "gtFine" / split / city / f"{base}_gtFine_labelIds.png"


def reduce_cityscapes_gt_for_eval(gt_seg: np.ndarray) -> np.ndarray:
    """Convert Cityscapes labelIds GT to trainIds 0..18 with 255 ignored."""
    gt = np.asarray(gt_seg)
    out = np.full(gt.shape, 255, dtype=np.int64)
    valid = (gt >= 0) & (gt < len(_CITYSCAPES_ID_TO_TRAIN_ID))
    out[valid] = _CITYSCAPES_ID_TO_TRAIN_ID[gt[valid].astype(np.int64)].astype(np.int64)
    return out


def reduce_cityscapes_pred_for_eval(pred_seg: np.ndarray) -> np.ndarray:
    """Convert internal CityScapes labels 1..19 to trainIds 0..18."""
    pred = np.asarray(pred_seg).astype(np.int64)
    out = np.full(pred.shape, 255, dtype=np.int64)
    valid = (pred >= 1) & (pred <= 19)
    out[valid] = pred[valid] - 1
    return out


def load_eval_samples(dataset_root: str) -> List[Dict[str, str]]:
    """Load validation samples for the configured dataset."""
    if DATASET_TYPE == "coco":
        from configs.config import COCO_IMAGELEVEL_VAL_JSON, COCO_VAL_LIST

        if Path(COCO_IMAGELEVEL_VAL_JSON).is_file():
            with open(COCO_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            sample_infos = []
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_filename = os.path.basename(img_info['file_name']).replace('.jpg', '.png')
                gt_split = 'train2014' if 'train' in img_info['file_name'] else 'val2014'
                gt_path = os.path.join(dataset_root, 'SegmentationClass', gt_split, gt_filename)
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        elif os.path.isfile(COCO_VAL_LIST):
            with open(COCO_VAL_LIST, 'r', encoding='utf-8') as f:
                val_ids = [line.strip() for line in f if line.strip()]
            sample_infos = []
            for img_stem in val_ids:
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                gt_path = os.path.join(dataset_root, 'SegmentationClass', 'val2014', f'{img_stem}.png')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': gt_path})
        else:
            segclass_dir = os.path.join(dataset_root, 'SegmentationClass', 'val2014')
            sample_infos = []
            for gt_file in sorted(Path(segclass_dir).glob('*.png')):
                img_stem = gt_file.stem
                img_path = os.path.join(dataset_root, 'images', 'val2014', f'{img_stem}.jpg')
                sample_infos.append({'id': img_stem, 'img': img_path, 'gt': str(gt_file)})
        print(f"Loaded {len(sample_infos)} COCO val images")
        return sample_infos

    if DATASET_TYPE == "ade20k":
        from configs.config import ADE20K_IMAGELEVEL_VAL_JSON

        sample_infos = []
        if Path(ADE20K_IMAGELEVEL_VAL_JSON).is_file():
            with open(ADE20K_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_path = os.path.join(
                    dataset_root,
                    'annotations',
                    'validation',
                    f"{Path(img_info['file_name']).stem}.png",
                )
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        else:
            image_dir = Path(dataset_root) / 'images' / 'validation'
            for img_file in sorted(image_dir.glob('*.jpg')):
                gt_path = Path(dataset_root) / 'annotations' / 'validation' / f'{img_file.stem}.png'
                sample_infos.append({'id': img_file.stem, 'img': str(img_file), 'gt': str(gt_path)})
        print(f"Loaded {len(sample_infos)} ADE20K val images")
        return sample_infos

    if DATASET_TYPE == "cityscapes":
        from configs.config import CITYSCAPES_IMAGELEVEL_VAL_JSON

        sample_infos = []
        if Path(CITYSCAPES_IMAGELEVEL_VAL_JSON).is_file():
            with open(CITYSCAPES_IMAGELEVEL_VAL_JSON, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            for img_info in val_data['images']:
                img_id = img_info['img_id']
                img_path = os.path.join(dataset_root, img_info['file_name'])
                gt_path = os.path.join(dataset_root, img_info['gt_file_name'])
                sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
        else:
            image_dir = Path(dataset_root) / 'leftImg8bit' / 'val'
            for img_file in sorted(image_dir.glob('*/*_leftImg8bit.png')):
                gt_path = cityscapes_image_to_gt_path(img_file, Path(dataset_root))
                sample_infos.append({'id': img_file.stem, 'img': str(img_file), 'gt': str(gt_path)})
        print(f"Loaded {len(sample_infos)} CityScapes val images")
        return sample_infos

    val_list = os.path.join(dataset_root, 'ImageSets', 'Segmentation', 'val.txt')
    gt_dir = os.path.join(dataset_root, 'SegmentationClassAug')
    img_dir = os.path.join(dataset_root, 'JPEGImages')

    with open(val_list, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]

    sample_infos = []
    for img_id in ids:
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        gt_path = os.path.join(gt_dir, f'{img_id}.png')
        sample_infos.append({'id': img_id, 'img': img_path, 'gt': gt_path})
    return sample_infos


def build_faiss_index(all_feats: np.ndarray):
    """Build the optional FAISS search index for gallery features."""
    if not INF_USE_FAISS:
        return None

    import faiss

    try:
        if hasattr(faiss, "set_random_seed"):
            faiss.set_random_seed(SEED)
        elif hasattr(faiss, "rand") and hasattr(faiss.rand, "seed"):
            faiss.rand.seed(SEED)
        elif hasattr(faiss, "cvar") and hasattr(faiss.cvar, "rand_seed"):
            faiss.cvar.rand_seed = int(SEED)
    except Exception:
        pass

    dim = all_feats.shape[1]
    n_samples = all_feats.shape[0]
    use_ivf = n_samples > INF_FAISS_EXACT_SEARCH_THRESHOLD

    if n_samples <= INF_FAISS_EXACT_SEARCH_THRESHOLD:
        print(f"[FAISS] {n_samples} features <= {INF_FAISS_EXACT_SEARCH_THRESHOLD}: IndexFlatIP (exact search)")

    if INF_FAISS_USE_GPU and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1536 * 1024 * 1024)

        if use_ivf:
            nlist = min(int(np.sqrt(n_samples)), 1024)
            print(f"[FAISS] GPU mode: IndexIVFFlat (nlist={nlist})")
            quantizer = faiss.IndexFlatIP(dim)
            cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            cpu_index.train(all_feats)
            cpu_index.add(all_feats)
            cpu_index.nprobe = min(32, nlist)
            return faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)

        print("[FAISS] GPU mode: IndexFlatIP (exact search)")
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(all_feats)
        return faiss.index_cpu_to_gpu(res, INF_FAISS_DEVICE, cpu_index)

    if use_ivf:
        nlist = min(int(np.sqrt(n_samples)), 1024)
        print(f"[FAISS] CPU mode: IndexIVFFlat (nlist={nlist})")
        faiss_index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim),
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        faiss_index.train(all_feats)
        faiss_index.add(all_feats)
        faiss_index.nprobe = min(32, nlist)
        return faiss_index

    print("[FAISS] CPU mode: IndexFlatIP (exact search)")
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(all_feats)
    return faiss_index


__all__ = [
    'build_faiss_index',
    'cityscapes_image_to_gt_path',
    'generate_proposals',
    'init_mask_proposer',
    'load_class_mapping',
    'load_class_features',
    'load_eval_samples',
    'load_proposals',
    'predict_instances',
    'rasterize_instances',
    'reduce_ade20k_gt_for_eval',
    'reduce_ade20k_pred_for_eval',
    'reduce_cityscapes_gt_for_eval',
    'reduce_cityscapes_pred_for_eval',
    'save_proposals',
]
