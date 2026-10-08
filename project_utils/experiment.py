"""Shared experiment I/O, migration lookup, immutable stages and training commands."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def relocated(value, root=None):
    """Resolve recorded paths via migration journals, without rewriting history."""
    root = ROOT if root is None else Path(root)
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    mappings = []
    for journal in sorted((root / 'storage/migrations').glob('*.jsonl')):
        for line in journal.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') == 'complete':
                mappings.append((Path(row['source']), Path(row['target'])))
    mappings.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    visited = set()
    while not path.exists() and path not in visited:
        visited.add(path)
        for source, target in mappings:
            if path == source or source in path.parents:
                path = target / path.relative_to(source)
                break
        else:
            if path.is_relative_to(root) and path.relative_to(root).parts and path.relative_to(root).parts[0] != 'experiments':
                archived = root / 'experiments' / path.relative_to(root)
                if archived.exists():
                    path = archived
            break
    if not path.exists():
        raise FileNotFoundError(f'Path unavailable after migration lookup: {value}; pass its actual path explicitly')
    return path.resolve()

def snapshot(paths):
    return [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(paths)]

def stage(output, key, resume, action):
    """Reuse complete stages; keep failed attempts and retry in fresh directories."""
    directory = output / key
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "stage_complete.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(marker)
        value = read_json(marker)
        paths = [Path(row["path"]) for row in value["artifacts"]]
        if snapshot(paths) != value["artifacts"]:
            raise ValueError(f"Completed stage outputs changed: {key}")
        print(f"[Resume] {key}", flush=True)
        return value["result"]
    index = 1
    while (directory / f"attempt_{index:03d}").exists():
        index += 1
    attempt = directory / f"attempt_{index:03d}"
    attempt.mkdir()
    print(f"[Stage] {key} -> {attempt}", flush=True)
    result, artifacts = action(attempt)
    write_json(marker, dict(result=result, artifacts=snapshot(artifacts)))
    return result

def train_command(raw, weights, bank, train_list, report_path, settings):
    command=[sys.executable,'-u',str(ROOT/'tools/train_feature_adapter.py'),
             '--source-bank',str(raw),'--train-list',str(train_list),
             '--output-dir',str(weights),'--export-bank',str(bank),'--device','cuda']
    if report_path:
        command += ['--split-file',str(report_path)]
    for key,value in settings.items():
        if value is not None:
            command += ['--'+key.replace('_','-'),str(value)]
    return command
