"""Resume a completed background extraction in a new version, without rewriting history.

Resolved symlink aliases are compared by path, size and SHA256, rather than list
order. The original launcher, workers, algorithms and signatures remain frozen.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_background_partition_queue as original

ab, gmp = original.ab, original.gmp
BaseStages = ab.Stages


def verified_artifacts(records):
    """Ignore only identical aliases/order; reject missing or changed bytes."""
    expected = {}
    for record in records:
        path = str(Path(record['path']).resolve())
        value = dict(record, path=path)
        if path in expected and expected[path] != value:
            raise ValueError('Conflicting artifact records: ' + path)
        expected[path] = value
    actual = {r['path']: r for r in ab.identities(expected)}
    if actual != expected:
        raise ValueError('Completed artifact content changed')


class AliasAwareStages(BaseStages):
    def verify_all(self):
        for marker in self.output.glob('*/complete.json'):
            verified_artifacts(ab.read(marker)['artifacts'])

    def run(self, key, action):
        marker = self.output / key / 'complete.json'
        if not marker.exists():
            return super().run(key, action)
        if not self.resume:
            raise FileExistsError(marker)
        value = ab.read(marker)
        verified_artifacts(value['artifacts'])
        print('[Resume verified] ' + key, flush=True)
        return copy.deepcopy(value['result'])


def verified_recovery(folder, active_classes):
    manifest = ab.read(folder / 'manifest.json')
    if (manifest['protocol'] != original.PROTOCOL
            or original.old.canonical(manifest['active']) != original.old.canonical(original.ACTIVE)
            or original.old.canonical(manifest['active_classes']) != original.old.canonical(active_classes)):
        raise ValueError('Recovery protocol/vocabulary differs')
    gmp.verify(manifest['code'], 'Unchanged background experiment source')
    gmp.verify(manifest['records'], 'Unchanged background experiment inputs')
    AliasAwareStages(folder, True).verify_all()
    mandatory = {'geometry', 'probe', 'extract_0', 'extract_1', 'merge', 'background_diagnostic'}
    allowed = mandatory | {f'eval_historical_{s}' for s in original.old.SEEDS}
    allowed |= {f'{stage}_{method}_{seed}' for stage in ('train', 'eval')
                for method in original.GROUPS for seed in original.old.SEEDS}
    completed = {}
    records = manifest['code'] + manifest['records'] + ab.identities([folder / 'manifest.json'])
    for marker in sorted(folder.glob('*/complete.json')):
        key = marker.parent.name
        if key not in allowed:
            raise ValueError('Unexpected recovery stage: ' + key)
        value = ab.read(marker)
        completed[key] = value
        records += ab.identities([marker]) + value['artifacts']
    if not mandatory.issubset(completed):
        raise ValueError('Recovery requires fully committed extraction, merge and diagnostic')
    gmp.verify(records, 'Recovery artifact identities')
    return records, completed


def main():
    if '--worker' in sys.argv:
        raise ValueError('Use the unchanged original worker entry')
    if '--recover-previous' not in sys.argv:
        raise ValueError('--recover-previous is required')
    previous_stages, previous_recovery = ab.Stages, original.verified_recovery
    ab.Stages = AliasAwareStages
    original.verified_recovery = verified_recovery
    try:
        original.main()
    finally:
        ab.Stages, original.verified_recovery = previous_stages, previous_recovery


if __name__ == '__main__':
    main()
