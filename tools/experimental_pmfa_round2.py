#!/usr/bin/env python3
"""Launch two independent, complete SMFA/PMFA ablations in the background."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_pmfa_ab as ab


def previous_inputs(folder):
    report = ab.read(folder/'comparison.json')
    manifest = ab.read(folder/'manifest.json')
    if report.get('status') != 'complete':
        raise ValueError('Previous comparison must be complete')
    if manifest.get('pmfa') != asdict(ab.PMFA()):
        raise ValueError('Expected previous PMFA default recipe; refusing a different starting point')
    if manifest.get('precision_policy') != ab.POLICY:
        raise ValueError('Previous precision policy differs')
    if ab.identities([entry['path'] for entry in manifest['inputs']]) != manifest['inputs']:
        raise ValueError('Previous input/weight hashes changed; refusing an incomparable round 2')
    common = []
    for key in ('dataset_root', 'image_dir', 'checkpoint', 'train_list', 'val_list', 'imagelevel_json', 'gt_dir'):
        value = manifest['options'].get(key)
        if not value:
            raise ValueError(f'Previous manifest lacks {key}')
        common += ['--'+key.replace('_', '-'), str(value)]
    for kind, flag in (('pseudo', '--pseudo-dir'), ('proposals', '--proposals-dir')):
        value = manifest['selected'].get(kind)
        if not value:
            value = ab.read(folder/kind/'complete.json')['result'][kind]
        if not Path(value).is_dir():
            raise FileNotFoundError(f'Previous shared input unavailable: {value}')
        common += [flag, str(value)]
    return common


def make_commands(common, tag):
    recipes = [('alpha025', 0, ['--alpha', '0.25']), ('steps0', 1, ['--steps', '0'])]
    return [(tag+'_'+name, [sys.executable, '-u', '-B', str(ab.SCRIPT),
             '--version', tag+'_'+name, '--gpu', str(gpu), *common, *change])
            for name, gpu, change in recipes]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, help='Previous complete experiment directory')
    parser.add_argument('--version', default='pmfa_round2_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if os.name != 'posix':
        parser.error('Run this background launcher on the Linux server')
    os.chdir(ROOT)
    ab.parse_args(['--version', args.version])  # Validate version before using it as a path.
    previous = args.previous
    if previous is None:
        candidates = []
        for report in sorted((ROOT/'experiments').glob('*/comparison.json')):
            try:
                if (ab.read(report).get('status') == 'complete' and
                    ab.read(report.parent/'manifest.json').get('pmfa') == asdict(ab.PMFA())):
                    candidates.append(report.parent)
            except (OSError, ValueError):
                continue
        if len(candidates) != 1:
            parser.error('Specify --previous; expected one complete default-PMFA experiment, found: '
                         + ', '.join(str(p) for p in candidates))
        previous = candidates[0]
    previous = previous.resolve()
    common = previous_inputs(previous)
    commands = make_commands(common, args.version)
    for version, command in commands:
        parsed = ab.parse_args(command[4:])
        context = ab.preflight(parsed)
        if not all(context['selected'].values()):
            raise ValueError('Previous caches are incomplete; round 2 refuses to regenerate separate inputs')
        for path in [ROOT/'experiments'/version, ROOT/'logs'/version,
                     ROOT/'weights/sam3_adapter'/version,
                     *(ROOT/'storage'/kind/'voc'/version for kind in
                       ('pseudo_labels', 'proposals', 'raw_banks', 'mlp_banks'))]:
            if path.exists():
                raise FileExistsError(f'Choose a fresh version: {path}')
    if args.preflight_only:
        print(json.dumps(dict(previous=str(previous), commands=commands), indent=2))
        return
    control = ROOT/'experiments'/args.version
    control.mkdir(parents=True, exist_ok=False)
    ab.write(control/'launch.json', dict(previous=str(previous), commands=commands,
                                        note='Each GPU runs fresh SMFA then its PMFA variant'))
    children = []
    try:
        for version, command in commands:
            log = ROOT/'logs'/f'{version}_launcher.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((version, process))
            ab.write(control/f'{version}.json', dict(pid=process.pid, log=str(log), command=command))
            print(f'Launched {version}: PID={process.pid}, log={log}', flush=True)
    except BaseException:
        for _, process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        raise
    print(f'Background launch complete (not experiment completion). PID records: {control}', flush=True)


if __name__ == '__main__':
    main()
