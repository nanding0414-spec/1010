# -*- coding: utf-8 -*-
"""
最底层诊断：直接看 SAM3 predict_inst 的原始输出（不过滤），定位"掩码太少"的根因。

跑法（ModuSeg_SAM3 根目录）：
    python tools/debug_predict_inst.py

对前 2 张 val 图，取第一/二批均匀点，打印：
  - logits / scores 的 shape 与量级（min/mean/max）
  - score 各阈值下命中数（0.88 / 0.5 / 0.1）
  - logits>0 的掩码面积分布（看大物体是否被覆盖）
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


def _grid(w, h, n):
    xs = (np.arange(n, dtype=np.float32) + 0.5) * w / n
    ys = (np.arange(n, dtype=np.float32) + 0.5) * h / n
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam3_model(
        device=device, checkpoint_path=SAM3_CKPT,
        enable_segmentation=False, enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model, device=str(device), confidence_threshold=0.0)

    for s in load_eval_samples(INF_DATASET_ROOT)[:2]:
        img = Image.open(s["img"]).convert("RGB")
        w, h = img.size
        state = processor.set_image(img)
        pts = _grid(w, h, 32)
        print(f"\n===== [{s['id']}] 原图 {w}x{h} =====")

        for bi, start in enumerate(range(0, 256, 64)):  # 只看前 256 个点，4 批
            batch = pts[start:start + 64]
            pc = batch[:, None, :]
            pl = np.ones((len(batch), 1), dtype=np.int64)
            logits, scores, _ = model.predict_inst(
                state, point_coords=pc, point_labels=pl,
                multimask_output=True, return_logits=True,
            )
            logits = np.asarray(logits)
            scores = np.asarray(scores)
            print(f"  batch{bi}: logits.shape={logits.shape}  scores.shape={scores.shape}")
            print(f"      logits  min/mean/max = {logits.min():.3f}/{logits.mean():.3f}/{logits.max():.3f}")
            print(f"      scores  min/mean/max = {scores.min():.4f}/{scores.mean():.4f}/{scores.max():.4f}")
            print(f"      scores>=0.88: {(scores >= 0.88).sum()}/{scores.size}  "
                  f">=0.5: {(scores >= 0.5).sum()}  >=0.1: {(scores >= 0.1).sum()}")
            masks = logits > 0.0
            areas = masks.sum(axis=(-1, -2))
            print(f"      掩码面积(logits>0) min/median/max = "
                  f"{areas.min():.0f}/{np.median(areas):.0f}/{areas.max():.0f}")
            if bi == 0:
                break  # 只看第一批就够了
    print("\n诊断参考：")
    print("  scores 最大若远小于 0.88 → pred_iou_thresh 全过滤，阈值设太高/量纲不对")
    print("  logits 若是 0~1 而非 ± 数值 → stability 公式失效(全 0)，需改 stability 过滤")
    print("  掩码面积 median 若很小 → 点提示没分割出物体(坐标/分辨率问题)")


if __name__ == "__main__":
    main()
