# -*- coding: utf-8 -*-
"""
快速诊断：定位 SAM3 推理 mIoU≈0 的原因。
跑在 ModuSeg_SAM3 根目录：
    python tools/debug_infer.py
只处理前 5 张 val 图，打印每张图的：
  - 提议器原始输出：proposal 数 / score 阈值命中数 / 面积命中数 / score 范围
  - 最终预测类别分布
据此判断：proposals≈0 或 score 很低 → 提议器问题；
          proposals 正常但全是 background → 检索/投票/特征问题。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OVERRIDE_DATASET_TYPE", "voc")

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from configs.config import (
    INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT, INF_NMS_IOU, INF_TOPK_NEIGH,
    INF_INSTANCE_SCORE_THR, INF_ALLOW_LEGACY_FEATURE_BANK, MIN_MASK_AREA, SEED,
    FEATURE_BACKBONE,
)
from project_utils.seed_utils import set_global_seed
from src.inference import (
    build_faiss_index, init_mask_proposer, load_class_features,
    load_class_mapping, load_eval_samples, predict_instances,
)
from src.feature_signature import validate_feature_signature
from src.retrieval import build_retrieval_context
from src.feature_extractor_factory import build_feature_extractor
from src.sam3_runtime import build_sam3_model


def main():
    set_global_seed(SEED)
    fb = Path(INF_FEATURE_BANK_ROOT)
    root = INF_DATASET_ROOT

    # ---- 特征库 + FAISS ----
    _, fg_names, bg_names = load_class_mapping(fb / "meta")
    ctf = load_class_features(
        fb / "features", list(dict.fromkeys(fg_names + bg_names))
    )
    gallery, labels = [], []
    gallery_dims = {}
    for name, mat in ctf.items():
        if mat is None or mat.shape[0] == 0:
            continue
        if mat.ndim != 2:
            raise RuntimeError(f"{name} feature matrix has invalid shape {mat.shape}")
        gallery_dims[name] = int(mat.shape[1])
        mat = mat.astype(np.float32)
        valid = np.all(np.isfinite(mat), axis=1) & (np.linalg.norm(mat, axis=1) >= 1e-6)
        mat = mat[valid]
        if mat.shape[0] == 0:
            continue
        gallery.append(mat)
        labels.extend([name] * mat.shape[0])
    if not gallery:
        raise RuntimeError("Feature bank is empty")
    if len(set(gallery_dims.values())) != 1:
        raise RuntimeError(f"Feature bank contains mixed dimensions: {gallery_dims}")
    all_feats = np.vstack(gallery)
    all_feats = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    all_feats = np.ascontiguousarray(all_feats.astype(np.float32))
    print(f"[diagnose] 特征库 {len(labels)} 条; 标签分布 Top5: "
          f"{Counter(labels).most_common(5)}")

    # ---- 模型 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = build_feature_extractor(device)
    validate_feature_signature(
        fb / "meta", dino.feature_signature(),
        allow_missing=INF_ALLOW_LEGACY_FEATURE_BANK,
    )
    if all_feats.shape[1] != dino.embed_dim:
        raise RuntimeError(
            f"Feature bank dim {all_feats.shape[1]} != extractor dim {dino.embed_dim}"
        )
    retrieval_context = build_retrieval_context(all_feats, labels)
    faiss_index = build_faiss_index(all_feats)
    sam3_model = build_sam3_model(
        device=device, enable_segmentation=False, enable_inst_interactivity=True
    )
    mask_proposer = init_mask_proposer(sam3_model=sam3_model)

    # ---- 前 5 张 val ----
    samples = load_eval_samples(root)[:5]
    for s in samples:
        img_bgr = cv2.imread(s["img"])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) 提议器原始输出
        preds = mask_proposer(img_bgr)
        masks = preds["instances"].pred_masks      # (N,H,W)
        scores = preds["instances"].scores         # (N,)
        n = int(masks.shape[0])
        n_thr = int((scores >= INF_INSTANCE_SCORE_THR).sum().item())
        areas = masks.float().sum(dim=(1, 2))
        n_area = int((areas >= MIN_MASK_AREA).sum().item())
        print(f"\n[{s['id']}] proposals={n}  score>={INF_INSTANCE_SCORE_THR}: {n_thr}  "
              f"area>={MIN_MASK_AREA}: {n_area}")
        if n > 0:
            print(f"          score[min/max/mean] = "
                  f"{scores.min().item():.3f}/{scores.max().item():.3f}/{scores.mean().item():.3f}")
            print(f"          mask面积[min/max] = "
                  f"{areas.min().item():.0f}/{areas.max().item():.0f}")

        # 2) 完整推理 → 类别分布
        inst = predict_instances(
            img_rgb, dino, mask_proposer, all_feats, labels,
            faiss_index, INF_NMS_IOU, topk_neigh=INF_TOPK_NEIGH,
            retrieval_context=retrieval_context,
        )
        print(f"          预测类别分布: {dict(Counter(i['cls_name'] for i in inst)) or '(空)'}")

    print("\n诊断结论参考：")
    print("  - proposals≈0 或 score 全部很低 -> 提议器(SAM3 均匀点提示)没产出有效掩码")
    print("  - proposals 正常但全是 background -> 检索投票被背景淹没 / 特征判别力不足")
    print("  - 出现部分前景类别 -> 只是分数阈值/网格密度/特征层的问题")


if __name__ == "__main__":
    main()
