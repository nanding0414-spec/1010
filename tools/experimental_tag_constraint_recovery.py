"""Recover the fixed tag-constraint experiment without altering its frozen source.

Bind inference environment to the exact job bank/checkpoint. Reuse only verified
completed training stages from the failed run; create a new experiment version.
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_tag_constraint_queue as original
ab, gmp, queue = original.ab, original.gmp, original.queue


def evaluation_environment(job, environment):
    """A cached extractor and the loaded bank must refer to the same model."""
    result = dict(environment)
    if job['stage'] == 'eval':
        result.update(
            OVERRIDE_FEATURE_ADAPTER_CKPT=str(Path(job['weights']) / 'adapter.pt'),
            OVERRIDE_INF_FEATURE_BANK_ROOT=job['bank'],
            OVERRIDE_INF_PROPOSALS_DIR=job['proposals'],
        )
    return result


def recovery_records(folder):
    manifest = ab.read(folder / 'manifest.json')
    if manifest['protocol'] != 'tag_constraint_v1':
        raise ValueError('Expected original tag-constraint experiment')
    gmp.verify(manifest['code'], 'Frozen recovery source')
    gmp.verify(manifest['records'], 'Frozen recovery inputs')
    ab.Stages(folder, True).verify_all()
    records = manifest['code'] + manifest['records'] + ab.identities([folder / 'manifest.json'])
    completed = {}
    for marker in sorted(folder.glob('train_*/complete.json')):
        value = ab.read(marker)
        result = value['result']
        job = ab.read(Path(result['attempt']) / 'job.json')
        method, seed = job['method'], job['seed']
        if method not in ('control', 'absent01') or seed not in original.old.SEEDS:
            raise ValueError('Unexpected recovery training stage')
        if job['coefficient'] != (0. if method == 'control' else original.COEFFICIENT):
            raise ValueError('Recovery loss differs')
        bank_sig = ab.read(Path(result['bank']) / 'meta/feature_signature.json')
        if bank_sig['adapter']['sha256'] != ab.digest(Path(result['weights']) / 'adapter.pt'):
            raise ValueError('Recovered bank/checkpoint mismatch')
        completed[marker.parent.name] = value
        records += ab.identities([marker]) + value['artifacts']
    gmp.verify(records, 'Recovery records')
    return records, completed


def stages_with_recovery(base, folder, completed):
    class RecoveryStages(base):
        def run(self, key, action):
            if (self.output / key / 'complete.json').exists() or key not in completed:
                return super().run(key, action)
            value = completed[key]
            gmp.verify(value['artifacts'], 'Recovered stage ' + key)
            marker = self.output / key / 'complete.json'
            # Record references, never copy or modify the old artifacts.
            ab.write(marker, dict(result=value['result'], artifacts=value['artifacts'],
                seconds=0., recovered_from=str(folder / key / 'complete.json'),
                historical_seconds=value['seconds']))
            print('[Recover read-only] ' + key, flush=True)
            return copy.deepcopy(value['result'])
    return RecoveryStages


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recover-previous', type=Path,
        default=ROOT / 'experiments/tag_constraint_20261005_124131')
    opts, remaining = parser.parse_known_args()
    if '--worker' in remaining:
        raise ValueError('Recovery launcher uses the unchanged original worker entry')
    folder = opts.recover_previous.resolve()
    records, completed = recovery_records(folder)
    print('[Recovery] exact input/source checks retained; new evaluation environment binding.', flush=True)
    originals = (queue.launch, ab.Stages, original.old.canonical, sys.argv)
    launch, stages, canonical, argv = originals
    def bound_launch(job, attempt, env, log, cancel, deadline):
        return launch(job, attempt, evaluation_environment(job, env), log, cancel, deadline)
    def identity(value):
        if isinstance(value, dict) and value.get('protocol') == 'tag_constraint_v1':
            value = dict(value, protocol='tag_constraint_recovery_v1',
                recovery=dict(previous=str(folder), records=records,
                    reused_training_stages=sorted(completed),
                    fix='bind evaluation bank/checkpoint/proposals environment to job'))
        return canonical(value)
    queue.launch = bound_launch
    ab.Stages = stages_with_recovery(stages, folder, completed)
    original.old.canonical = identity
    sys.argv = [__file__, *remaining]
    try:
        original.main()
        gmp.verify(records, 'Recovery records after execution')
    finally:
        queue.launch, ab.Stages, original.old.canonical, sys.argv = originals


if __name__ == '__main__':
    main()
