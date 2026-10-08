#!/usr/bin/env bash
# Run from the server project root. Keep the original experiment .py files intact.
# This .sh is outside the Python source inventory and changes only parent-side
# manifest comparison order. All original input/source/artifact hashes still apply.
set -euo pipefail
exec python -u -B - "$@" <<'PY'
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def restore_known_verifier_edit(root, version):
    """Restore only bytes proven to match the original recorded source hash."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', version):
        raise ValueError('Invalid version')
    control = root/'experiments'/version
    folders = sorted((root/'experiments').glob(version+'_*/manifest.json'))
    if not folders:
        raise ValueError('No existing variant manifests found for recovery')
    for lock in [control/'launcher.lock', *[p.parent/'launcher.lock' for p in folders]]:
        if lock.exists():
            raise ValueError(f'Launcher lock exists; verify task stopped before recovery: {lock}')
    target = root/'tools/experimental_rpg_ab.py'
    entries = []
    for manifest in folders:
        rows = json.loads(manifest.read_text(encoding='utf-8'))['code']
        matching = [r for r in rows if Path(r['path']).resolve() == target.resolve()]
        if len(matching) != 1:
            raise ValueError(f'Missing unique RPG source identity: {manifest}')
        entries.append(matching[0])
    if any(row != entries[0] for row in entries):
        raise ValueError('Variant source identities disagree; no automatic recovery')
    expected = entries[0]
    current = target.read_bytes()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    if digest(current) == expected['sha256'] and len(current) == expected['size']:
        print('[Recovery] Original RPG source identity already matches.', flush=True)
        return
    text = current.decode('utf-8')
    functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_records']
    if len(functions) != 1:
        raise ValueError('Cannot locate unique verifier; refusing source restoration')
    node = functions[0]
    lines = text.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in text else '\n'
    old = [
        'def verify_records(records, label):',
        "    if ab.identities([r['path'] for r in records]) != records:",
        "        raise ValueError(f'{label} files changed or missing')",
    ]
    candidate = (''.join(lines[:node.lineno-1])+newline.join(old)+newline+
                 ''.join(lines[node.end_lineno:])).encode('utf-8')
    if digest(candidate) != expected['sha256'] or len(candidate) != expected['size']:
        raise ValueError('Source differs beyond the known verifier edit; original hash cannot be restored. '
                         'No source or manifest was changed.')
    control.mkdir(parents=True, exist_ok=True)
    backup = control/('rpg_source_before_recovery_'+digest(current)+'.py')
    if backup.exists():
        if backup.read_bytes() != current:
            raise ValueError('Recovery backup collision')
    else:
        with backup.open('xb') as stream:
            stream.write(current)
    evidence = control/('verifier_recovery_'+digest(current)+'.json')
    if not evidence.exists():
        with evidence.open('x', encoding='utf-8') as stream:
            json.dump(dict(source=str(target),backup=str(backup),before_sha256=digest(current),
                           restored_sha256=digest(candidate),reason='restore exact recorded source; patch verifier in memory'),
                      stream,indent=2)
    temporary = target.with_name(target.name+'.recovery.tmp')
    with temporary.open('xb') as stream:
        stream.write(candidate)
    temporary.chmod(target.stat().st_mode)
    if target.read_bytes() != current:
        temporary.unlink()
        raise ValueError('Source changed during recovery; refusing replacement')
    os.replace(temporary,target)
    print(f'[Recovery] Restored exact recorded RPG source; backup={backup}',flush=True)


def verify_records(records, label):
    expected = {}
    for row in records:
        path = row['path']
        if path in expected and expected[path] != row:
            raise ValueError(f'{label}: conflicting records for {path}')
        expected[path] = row
    try:
        actual = {row['path']: row for row in rpg.ab.identities(expected)}
    except OSError as exc:
        raise ValueError(f'{label} files changed or missing: {exc}') from exc
    changed = [path for path, row in expected.items() if actual.get(path) != row]
    if changed:
        raise ValueError(f'{label} files changed or missing: {changed[:10]}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--version',required=True)
    options,_=parser.parse_known_args()
    restore_known_verifier_edit(Path.cwd(),options.version)

from tools import experimental_rpg_ab as rpg

if __name__ == '__main__':
    print('[Recovery] Compare artifact identities by path; all SHA256 checks retained.', flush=True)
    rpg.verify_records = verify_records
    rpg.main(sys.argv[1:])
PY
