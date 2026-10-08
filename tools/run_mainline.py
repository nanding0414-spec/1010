#!/usr/bin/env python3
"""Strict original-image GMP/MLP2/HCR inference (no multiview inputs)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True, type=Path)
    p.add_argument('--adapter', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'weights/sam3/sam3.pt')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--clean-proposals-dir', type=Path, required=True)
    p.add_argument('--raw-proposals-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--seed', type=int, choices=[42,123,2026], default=42)
    p.add_argument('--no-hcr', action='store_true', help='Clean-baseline ablation only')
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory; historical results are never overwritten')
    for item in (a.bank/'meta/feature_signature.json', a.adapter, a.checkpoint,
                 a.clean_proposals_dir/'cleanup_config.json'):
        if not item.is_file():
            p.error(f'Missing required input: {item}')
    signature = json.loads((a.bank/'meta/feature_signature.json').read_text())
    from src.hard_center_gmp import describe
    arch = signature.get('adapter', {}).get('architecture', {})
    if (signature.get('aggregation') != describe(.1) or arch.get('num_layers', 2) != 2
            or (arch.get('input_dim'), arch.get('hidden_dim'), arch.get('output_dim')) != (1280,512,256)):
        p.error('Mainline requires a native hard-center GMP rho=.1 bank and plain MLP1280→512→256')
    try:
        gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        occupancy = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
                     for row in gpu_rows.strip().splitlines()}
        if a.gpu not in occupancy or occupancy[a.gpu][0] > 1000 or occupancy[a.gpu][1] > 10:
            p.error(f'Physical GPU {a.gpu} is absent or occupied: {occupancy.get(a.gpu)}')
    except (OSError, subprocess.CalledProcessError) as exc:
        p.error(f'Cannot inspect GPU occupancy: {exc}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONDONTWRITEBYTECODE='1',
        OVERRIDE_DATASET_TYPE='voc', OVERRIDE_SEED=str(a.seed), OVERRIDE_SAM3_POOLING='hard_center_gmp',
        OVERRIDE_SAM3_GMP_RHO='0.1', OVERRIDE_SAM3_USE_BF16='false',
        OVERRIDE_SAM3_RESOLUTION='1008', OVERRIDE_SAM3_FEATURE_SOURCE='trunk_fpn',
        OVERRIDE_SAM3_FEATURE_LEVEL='-1', OVERRIDE_SAM3_TOKEN_NORM='layernorm',
        OVERRIDE_SAM3_CONTEXT_SUBTRACT='0', OVERRIDE_INF_LOAD_PROPOSALS='true',
        OVERRIDE_INF_FEATURE_BANK_ROOT=str(a.bank.resolve()),
        OVERRIDE_FEATURE_ADAPTER_CKPT=str(a.adapter.resolve()),
        OVERRIDE_SAM3_CKPT=str(a.checkpoint.resolve()),
        OVERRIDE_INF_PROPOSALS_DIR=str(a.clean_proposals_dir.resolve()),
        OVERRIDE_INF_DATASET_ROOT=str(a.dataset_root.resolve()),
        OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK='false',
        OVERRIDE_INF_RETRIEVAL_MODE='balanced', OVERRIDE_INF_TOPK_NEIGH='25',
        OVERRIDE_INF_CLASS_TOPM='5', OVERRIDE_INF_RETRIEVAL_SEARCH_MULTIPLIER='2',
        OVERRIDE_INF_CENTROID_WEIGHT='0.4', OVERRIDE_INF_RETRIEVAL_MARGIN_THR='0',
        OVERRIDE_INF_PROPOSAL_SCORE_WEIGHT='0.2', OVERRIDE_INF_NMS_IOU='0.4',
        OVERRIDE_INF_INSTANCE_SCORE_THR='0.02')
    command = [sys.executable, '-B', str(ROOT/'run_inference.py'), '--strict-eval',
        '--output-dir', str(a.output_dir.resolve()), '--results-json', str(a.output_dir.resolve()/'metrics.json')]
    if not a.no_hcr:
        command += ['--rescue', '--rescue-raw-dir', str(a.raw_proposals_dir.resolve())]
    started = __import__('time').monotonic()
    status = 'failed'
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        status = 'complete'
    finally:
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(a.output_dir.name, status, str(a.output_dir), __import__('time').monotonic()-started)


if __name__ == '__main__':
    main()
