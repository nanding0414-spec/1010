#!/usr/bin/env python3
"""Stage 1 of split inference: generate SAM3 proposals only.

Runs the SAM3 point-prompt mask proposer on every validation image and
saves per-image proposal masks + scores to disk as npz files. Later,
run_inference.py with OVERRIDE_INF_LOAD_PROPOSALS=true will load these
precomputed proposals and skip the SAM3 proposal stage, running only
feature extraction -> FAISS retrieval -> voting -> NMS -> rasterize -> mIoU.

Usage:
    conda activate moduseg_sam3
    export CC=$(which gcc)
    OVERRIDE_DATASET_TYPE=voc python run_generate_proposals.py
"""
import os
import time
import argparse

import cv2
import torch
from tqdm import tqdm

from configs.config import (
    DATASET_TYPE, INF_DATASET_ROOT, INF_FEATURE_BANK_ROOT, INF_PROPOSALS_DIR,
    INF_MASK_BACKEND, SEED, SAM3_CKPT, SAM3_QUERY_PROMPTS,
)
from configs.config_helpers import get_model_version_str
from project_utils.config_logger import record_config
from project_utils.seed_utils import set_global_seed
from src.inference import (
    generate_proposals, init_mask_proposer, load_eval_samples, save_proposals,
)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("=" * 80)
    print(f"Generate Proposals (Stage 1) - {DATASET_TYPE.upper()}")
    print("=" * 80)

    set_global_seed(SEED)

    sample_infos = load_eval_samples(INF_DATASET_ROOT)
    proposals_dir = os.path.abspath(INF_PROPOSALS_DIR)
    os.makedirs(proposals_dir, exist_ok=False)
    print(f"[Proposals] output dir: {proposals_dir}")
    print(f"[Proposals] {len(sample_infos)} validation images")

    # This stage only creates masks. The feature model is selected separately
    # by run_build_feature_bank.py/run_inference.py.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from src.sam3_runtime import build_sam3_model
    use_query_backend = INF_MASK_BACKEND == "sam3_query"
    dino = build_sam3_model(
        device=device,
        checkpoint_path=SAM3_CKPT,
        enable_segmentation=use_query_backend,
        enable_inst_interactivity=not use_query_backend,
    )
    model_version = get_model_version_str("sam3")

    mask_proposer = init_mask_proposer(sam3_model=dino)
    print(f"[Proposals] mask backend: {INF_MASK_BACKEND}")
    if use_query_backend:
        print(f"[Proposals] query prompts: {SAM3_QUERY_PROMPTS}")
    record_config(os.path.join(proposals_dir, 'meta'), 'run_generate_proposals.py')

    n_processed = 0
    n_skipped = 0
    t0 = time.time()
    for sample in tqdm(sample_infos, desc=f"Propose ({model_version})", ncols=100):
        img_id = sample['id']
        img_path = sample['img']

        # Skip if already generated
        out_path = os.path.join(proposals_dir, f"{img_id}.npz")
        if os.path.isfile(out_path):
            n_skipped += 1
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'Cannot read validation image: {img_path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        masks, scores = generate_proposals(img_rgb, mask_proposer)
        save_proposals(proposals_dir, img_id, masks, scores)
        n_processed += 1

    elapsed = time.time() - t0
    print(f"\n[Done] processed={n_processed}, skipped(exists)={n_skipped}, "
          f"total={len(sample_infos)}, time={elapsed:.1f}s")
    print(f"[Done] proposals saved to: {proposals_dir}")
    print(f"\nNext step: run inference with precomputed proposals:")
    print(
        f"  OVERRIDE_DATASET_TYPE={DATASET_TYPE} "
        f"OVERRIDE_INF_MASK_BACKEND={INF_MASK_BACKEND} "
        f"OVERRIDE_INF_LOAD_PROPOSALS=true "
        f"OVERRIDE_INF_PROPOSALS_DIR={INF_PROPOSALS_DIR} python run_inference.py"
    )


if __name__ == '__main__':
    main()
