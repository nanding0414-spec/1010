# -*- coding: utf-8 -*-
"""
逐阶段过滤诊断：看掩码到底在哪一步被丢掉。

    python tools/debug_mask_stages.py

对第 1 张 val 图，跑全部 1024 个点，打印每一步剩余的掩码数：
  原始(logits>0) -> score 阈值 -> stability 阈值 -> box NMS -> 最终
并打印 top 掩码面积，判断是否有"超大背景块"在 NMS 里压掉其它掩码。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OVERRIDE_DATASET_TYPE", "voc")

import numpy as np
import torch
from PIL import Image

from configs.config import (
    INF_DATASET_ROOT, SAM3_CKPT,
)
from sam3.model.sam3_image_processor import Sam3Processor
from src.inference import load_eval_samples
from src.sam3_runtime import build_sam3_model
from src.mask_proposer_sam3 import (
    _make_point_grid, _stability_score, _mask_bbox, _nms_by_boxes,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam3_model = build_sam3_model(
        device=device, checkpoint_path=SAM3_CKPT,
        enable_segmentation=False, enable_inst_interactivity=True,
    )
    processor = Sam3Processor(sam3_model, device=str(device), confidence_threshold=0.0)
    s = load_eval_samples(INF_DATASET_ROOT)[0]
    img = Image.open(s["img"]).convert("RGB")
    w, h = img.size
    state = processor.set_image(img)
    pts = _make_point_grid(w, h, 32)

    logits_all, scores_all = [], []
    for b in range(0, len(pts), 64):
        batch = pts[b:b + 64]
        pc = batch[:, None, :]
        pl = np.ones((len(batch), 1), dtype=np.int64)
        logits, scores, _ = sam3_model.predict_inst(
            state, point_coords=pc, point_labels=pl,
            multimask_output=True, return_logits=True,
        )
        logits = np.asarray(logits)
        scores = np.asarray(scores)
        if logits.ndim == 3:
            logits = logits[None, ...]
            scores = scores[None, ...]
        for bi in range(logits.shape[0]):
            for ci in range(logits.shape[1]):
                logits_all.append(logits[bi, ci])
                scores_all.append(float(scores[bi, ci]))

    N = len(logits_all)
    print(f"[{s['id']}] 原始掩码数(logits>0): {N}")
    areas = np.array([int((l > 0.0).sum()) for l in logits_all])
    print(f"  面积分布: min={areas.min()}  p25={np.percentile(areas,25):.0f}  "
          f"median={np.percentile(areas,50):.0f}  p75={np.percentile(areas,75):.0f}  max={areas.max()}")

    for thr in (0.5, 0.7, 0.88):
        print(f"  score >= {thr}: {sum(1 for x in scores_all if x >= thr)}/{N}")

    # stability（mask_threshold=0, offset=1.0）
    for thr in (0.5, 0.7, 0.95):
        cnt = sum(
            1 for l in logits_all if _stability_score(l, 0.0, 1.0) >= thr
        )
        print(f"  stability >= {thr}: {cnt}/{N}")

    # 完整串联：score>=0.5 + stability>=0.7
    keep_masks, keep_scores = [], []
    for l, sc in zip(logits_all, scores_all):
        if sc < 0.5:
            continue
        if _stability_score(l, 0.0, 1.0) < 0.7:
            continue
        m = l > 0.0
        if m.sum() < 300:
            continue
        keep_masks.append(m)
        keep_scores.append(sc)
    print(f"\n  score>=0.5 & stab>=0.7 & area>=300: {len(keep_masks)}")
    for thr in (0.7, 0.85, 0.95):
        print(f"    再经 box NMS {thr}: {len(_nms_by_boxes(keep_masks, keep_scores, thr))}")

    # top 面积，看是否有超大块
    order = np.argsort(-areas)
    top = areas[order][:10]
    print(f"\n  top10 掩码面积: {top.tolist()}  (若存在接近全图面积的超大块，说明需 max_mask_area_ratio 剔除)")
    print(f"  全图面积 = {w*h}")


if __name__ == "__main__":
    main()
