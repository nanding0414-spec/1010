"""Mainline semantic-label geometry cleanup: immutable source, checked per-image outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from configs.experiment import PSEUDO_CLEANUP
    from src.proposal_cleanup import CleanupConfig
    from src.pseudo_label_cleanup import POLICY, clean_label_file
    from tools.run_mainline_full import read, write, free_space, sha
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--image-list', type=Path, required=True)
    for name, value in PSEUDO_CLEANUP.items():
        p.add_argument('--'+name.replace('_','-'), type=float if isinstance(value,float) else int, default=value)
    a=p.parse_args(argv);source=a.input_dir.resolve();output=a.output_dir.resolve()
    if source==output or source in output.parents or output in source.parents:
        p.error('Input and output label directories must be separate')
    ids=a.image_list.read_text().split()
    if not ids or len(ids)!=len(set(ids)) or any(Path(i).name!=i or i in ['.','..'] for i in ids):
        p.error('Image list must contain unique safe stems')
    config=CleanupConfig(**{k:getattr(a,k) for k in PSEUDO_CLEANUP})
    code={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'src/pseudo_label_cleanup.py',ROOT/'src/proposal_cleanup.py']}
    digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    settings=dict(input_dir=str(source),output_dir=str(output),image_ids=ids,policy=POLICY,
                  geometry=PSEUDO_CLEANUP|{k:getattr(a,k) for k in PSEUDO_CLEANUP},code_sha256=code)
    metadata=output/'cleanup_config.json'
    if metadata.exists():
        if read(metadata)['settings']!=settings:raise ValueError('Cleanup source, settings or code changed; choose a new output')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Existing labels have no cleanup receipt; choose a new output')
    output.mkdir(parents=True,exist_ok=True)
    write(metadata,dict(settings=settings,status='running'))
    stats={k:0 for k in ['changed_images','changed_pixels','removed_island_pixels','proposed_hole_pixels',
                        'filled_background_pixels','protected_other_class_pixels','area_rejected_classes']}
    for index,stem in enumerate(ids,1):
        free_space(output)
        row=clean_label_file(source/(stem+'.png'),output/(stem+'.png'),config,digest)
        stats['changed_images']+=int(row['changed'])
        for key in stats:
            if key!='changed_images':stats[key]+=row[key]
        print(f'[Semantic cleanup] {index}/{len(ids)} {stem}',flush=True)
    write(metadata,dict(settings=settings,status='complete',count=len(ids),statistics=stats))
    return 0


if __name__=='__main__':raise SystemExit(main())
