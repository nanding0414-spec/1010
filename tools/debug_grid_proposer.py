# -*- coding: utf-8 -*-
"""
调试：可视化 SAM3 均匀点提示（SAM3GridProposer）产出的原始提议掩码。

跑在 ModuSeg_SAM3 根目录：
    python tools/debug_grid_proposer.py [--grid 32] [--num 5] [--dataset voc]

对前 NUM 张 val 图：
  1) 打印每张图的：proposals 数 / score 范围与均值 / 掩码面积范围
  2) 保存两类可视化到当前特征库的 grid_proposals_debug/：
       {id}_all.png    所有网格点提示掩码（随机色叠加，看覆盖范围）
       {id}_thr.png    仅 score>=阈值 的掩码（看真正能进入推理的）

如果 all 图基本是空/碎渣、或 thr 图基本没有东西 → SAM3 均匀点提示没产出有效物体掩码。
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OVERRIDE_DATASET_TYPE", "voc")

from pathlib import Path

import cv2
import numpy as np
import torch

from configs.config import (
    INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT, INF_INSTANCE_SCORE_THR, MIN_MASK_AREA,
    SAM3_CKPT,
)
from src.inference import load_eval_samples
from src.mask_proposer_sam3 import SAM3GridProposer
from src.sam3_runtime import build_sam3_model


def _overlay(img, masks, scores=None, thr=None):
    """把所有掩码用随机色叠加到原图上，返回 overlay 图。"""
    h, w = img.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    n = masks.shape[0]
    for i in range(n):
        m = masks[i].astype(bool)
        if m.sum() < 10:
            continue
        if thr is not None and scores[i] < thr:
            continue
        colored[m] = rng.randint(40, 255, 3)
    return cv2.addWeighted(img, 0.6, colored, 0.4, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32, help="SAM3 均匀点网格大小")
    ap.add_argument("--crop", type=int, default=0, help="多尺度 crop 层数(0/1/2)")
    ap.add_argument("--num", type=int, default=5, help="处理前几张 val 图")
    ap.add_argument("--dataset", type=str, default="voc")
    args = ap.parse_args()
    os.environ["OVERRIDE_DATASET_TYPE"] = args.dataset
    os.environ["OVERRIDE_SAM3_GRID_SIZE"] = str(args.grid)

    out = Path(INF_FEATURE_BANK_ROOT) / "grid_proposals_debug"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam3_model = build_sam3_model(
        device=device, checkpoint_path=SAM3_CKPT,
        enable_segmentation=False, enable_inst_interactivity=True,
    )
    proposer = SAM3GridProposer(
        model=sam3_model, grid_size=args.grid, crop_n_layers=args.crop,
    )

    samples = load_eval_samples(INF_DATASET_ROOT)[: args.num]
    print(f"处理 {len(samples)} 张 val 图, 网格 {args.grid}x{args.grid} "
          f"({args.grid*args.grid} 个点提示), crop层数={args.crop}, 输出到 {out}/")

    for s in samples:
        img = cv2.imread(s["img"])
        if img is None:
            continue
        preds = proposer(img)
        masks = preds["instances"].pred_masks.detach().cpu().numpy()  # (N,H,W)
        scores = preds["instances"].scores.detach().cpu().numpy()     # (N,)
        n = masks.shape[0]
        areas = masks.sum(axis=(1, 2))
        print(f"\n[{s['id']}] proposals={n}  "
              f"score min/mean/max = {scores.min():.3f}/{scores.mean():.3f}/{scores.max():.3f}")
        if n > 0:
            print(f"    score>=INF_INSTANCE_SCORE_THR({INF_INSTANCE_SCORE_THR}): "
                  f"{int((scores >= INF_INSTANCE_SCORE_THR).sum())}  "
                  f"area>={MIN_MASK_AREA}: {int((areas >= MIN_MASK_AREA).sum())}")
            print(f"    mask面积 min/max = {areas.min():.0f}/{areas.max():.0f}")

        cv2.imwrite(str(out / f"{s['id']}_all.png"), _overlay(img, masks))
        cv2.imwrite(
            str(out / f"{s['id']}_thr.png"),
            _overlay(img, masks, scores, thr=INF_INSTANCE_SCORE_THR),
        )

    print(f"\n完成，可视化已存到 {out}/。")
    print("判断：_all 基本空/碎渣 或 _thr 几乎没有东西 → 均匀点提示没产出有效物体掩码，"
          "问题在提议器(SAM3 提示策略)。")


if __name__ == "__main__":
    main()
