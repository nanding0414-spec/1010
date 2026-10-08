#!/usr/bin/env python3
"""Archive completed experiments out of the root. Default: print plan only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

NAMES = (
    'retrieval_sweep_mlp3_score07', 'results_active_v1_val',
    'pooling_ablation_v1', 'feature_bank_vis_pooling_v1',
    'active_v1_smfa_mlp3_best', 'cleanup_sweep_voc_v1',
    'cleanup_sweep_voc_v2', 'cleanup_sweep_voc_v3', 'cleanup_sweep_voc_v4',
)


def plan(root):
    rows = []
    for name in NAMES:
        source = root / name
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink():
            raise ValueError(f'Inspect root symlink manually before archiving: {source}')
        if not source.is_dir():
            raise ValueError(f'Expected directory: {source}')
        target = root / 'experiments' / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        # Relative links could change their meaning at the new nesting depth.
        for base, dirs, files in os.walk(source, followlinks=False):
            for child in dirs + files:
                path = Path(base) / child
                if path.is_symlink() and not os.path.isabs(os.readlink(path)):
                    raise ValueError(f'Relative symlink needs manual review: {path}')
        rows.append({'source': str(source), 'target': str(target)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--confirm-idle', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'run_inference.py').is_file():
        parser.error('Not a project root')
    rows = plan(root)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print('No old-root links will be created. Historical JSON paths are preserved as records; old resume commands may no longer work.')
    if not args.apply:
        print('[Read only] Add --apply --confirm-idle after all jobs finish.')
        return
    if sys.platform != 'linux' or not args.confirm_idle:
        parser.error('Apply requires Linux and --confirm-idle')
    destination = root / 'experiments'
    if destination.is_symlink():
        parser.error('experiments must not be a symlink')
    destination.mkdir(exist_ok=True)
    for row in rows:
        if Path(row['source']).stat().st_dev != destination.stat().st_dev:
            parser.error('Cross-filesystem archive is not supported')
    journals = root / 'storage/migrations'
    journals.mkdir(parents=True, exist_ok=True)
    journal = journals / f'archive_{time.time_ns()}.jsonl'
    with journal.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(dict(event='intent', **row)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            Path(row['source']).rename(row['target'])
            stream.write(json.dumps(dict(event='complete', **row)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print('[Archived]', row['target'])
    print('[Journal]', journal)


if __name__ == '__main__':
    main()
