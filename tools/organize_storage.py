#!/usr/bin/env python3
"""Inventory or relocate legacy VOC artifacts, retaining old paths as symlinks.

Default is read-only. Apply only on an idle Linux server. No data is deleted.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'sam3-main', '.git', '__pycache__', 'src', 'tools', 'tests',
        'configs', 'project_utils', 'weights', 'artifacts', 'logs', 'storage'}


def discover(root, extra_proposals=()):
    candidates = {}
    pseudo = root / 'data/VOC2012/pseudo'
    if pseudo.is_dir():
        for path in pseudo.iterdir():
            if path.is_dir() and not path.is_symlink():
                candidates[path] = 'pseudo_labels'
    for path in root.glob('proposals*'):
        if path.is_dir() and not path.is_symlink():
            candidates[path] = 'proposals'
    for base, dirs, files in os.walk(root, followlinks=False):
        directory = Path(base)
        dirs[:] = [d for d in dirs if d not in SKIP and d != 'data'
                   and not (directory / d).is_symlink()]
        if directory.name == 'meta' and 'feature_signature.json' in files:
            bank = directory.parent
            if not (bank / 'features').is_dir():
                continue
            signature = json.loads((directory / 'feature_signature.json').read_text(encoding='utf-8'))
            candidates[bank] = 'mlp_banks' if signature.get('adapter') else 'raw_banks'
        # Include cleaned proposal exports inside experiments as well.
        for name in list(dirs):
            if name in {'best_proposals', 'cleaned_proposals'}:
                candidates[directory / name] = 'proposals'
                dirs.remove(name)
    for value in extra_proposals:
        path = Path(os.path.abspath(value))
        if not path.is_dir():
            raise ValueError(f'Proposal directory missing: {path}')
        if not path.is_symlink():
            candidates[path] = 'proposals'
    rows = []
    for path, category in sorted(candidates.items()):
        if path.is_relative_to(root / 'storage'):
            continue
        if category == 'pseudo_labels':
            name = path.name
        elif path.is_relative_to(root):
            name = '__'.join(path.relative_to(root).parts)
        else:
            name = 'external__' + path.parent.name + '__' + path.name
        target = root / 'storage' / category / 'voc' / name
        rows.append(dict(category=category, source=str(path), target=str(target)))
    return rows


def migrate(rows, journal):
    # Validate the entire plan before the first rename.
    sources = [Path(row['source']) for row in rows]
    targets = [Path(row['target']) for row in rows]
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate destinations')
    for source, target in zip(sources, targets):
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f'Source changed: {source}')
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        if any(source != other and source.is_relative_to(other) for other in sources):
            raise ValueError(f'Nested migration sources: {source}')
        ancestor = target.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if source.stat().st_dev != ancestor.stat().st_dev:
            raise ValueError(f'Cross-filesystem move requires separate copy/verification: {source}')
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x', encoding='utf-8') as log:
        for row, source, target in zip(rows, sources, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            log.write(json.dumps(dict(event='intent', **row), ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            source.rename(target)
            try:
                source.symlink_to(target, target_is_directory=True)
            except BaseException:
                target.rename(source)
                raise
            log.write(json.dumps(dict(event='complete', **row), ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            print(f'[Moved + linked] {source} -> {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--include-proposals', type=Path, action='append', default=[])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--confirm-idle', action='store_true',
                        help='Operator confirms no build/train/eval/generation jobs are running')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'run_build_feature_bank.py').is_file():
        parser.error('Root is not a ModuSeg project')
    rows = discover(root, args.include_proposals)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f'[Plan] {len(rows)} directories; data contents remain unchanged', flush=True)
    if args.apply:
        if sys.platform != 'linux' or not args.confirm_idle:
            parser.error('Apply requires Linux and --confirm-idle; stop/finish all jobs first')
        journal = root / 'storage/migrations' / f'{time.time_ns()}.jsonl'
        migrate(rows, journal)
        print(f'[Journal] {journal}')
    else:
        print('[Read-only] No directories moved. Review the plan before --apply --confirm-idle.')


if __name__ == '__main__':
    main()
