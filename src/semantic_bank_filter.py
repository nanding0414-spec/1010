"""Non-destructive post-LSA gallery filtering for the published mainline."""
from pathlib import Path
import json
import shutil
import numpy as np


def select_members(features, sample_ids, *, background=False):
    # Reuse the exact FP32 distance/floor/tie/retained-order implementation
    # independently accepted in the fixed postfilter experiment.
    from tools.bank_postfilter_sweep_core import trim
    return trim(features, sample_ids, 5, background)


def export_filtered_bank(source, target):
    from src.feature_bank_storage import load_feature_matrix_npz, save_feature_matrix_npz
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or target == source:
        raise FileExistsError('Postfilter needs a new directory; source bank stays immutable')
    if (source/'meta/postfilter.json').exists():
        raise ValueError('This bank is already postfiltered; filtering twice is prohibited')
    signature = json.loads((source/'meta/feature_signature.json').read_text())
    if not signature.get('adapter'):
        raise ValueError('Postfilter applies to an exported LSA bank, never raw HC-GMP vectors')
    mapping = json.loads((source/'meta/class_mapping.json').read_text())['id_to_dirname']
    if set(map(int, mapping)) != set(range(21)):
        raise ValueError('Expected all 21 VOC classes, including background')
    target.mkdir(parents=True)
    shutil.copytree(source/'meta', target/'meta')
    classes = []
    for cid, name in sorted(mapping.items(), key=lambda row: row[1]):
        files = list((source/'features'/name).glob('*.npz'))
        if len(files) != 1:
            raise ValueError(f'Expected one matrix for {name}')
        x, ids = load_feature_matrix_npz(files[0])
        if not len(x):
            raise ValueError(f'Empty mainline class {name}')
        keep, record = select_members(x, ids, background=int(cid) == 0)
        save_feature_matrix_npz(target/'features'/name/files[0].name, x[keep], ids[keep])
        classes.append(dict(record, class_id=int(cid), name=name))
    report = dict(recipe_id='lsa_hcgmp_prr_v1', source_bank=str(source),
                  percent=5, background_deleted=0, classes=classes,
                  total_kept=sum(c['kept'] for c in classes), query_filtering=False)
    (target/'meta/postfilter.json').write_text(json.dumps(report, indent=2))
    return report


def validate_filtered_bank(bank):
    from src.feature_bank_storage import load_feature_matrix_npz
    bank=Path(bank)
    report=json.loads((bank/'meta/postfilter.json').read_text())
    if report.get('recipe_id')!='lsa_hcgmp_prr_v1' or report.get('percent')!=5 or report.get('background_deleted')!=0:
        raise ValueError('Expected the fixed post-LSA FG5% bank')
    mapping=json.loads((bank/'meta/class_mapping.json').read_text())['id_to_dirname']
    rows=report['classes']
    if {r['class_id'] for r in rows}!=set(range(21)) or len(rows)!=21:
        raise ValueError('Incomplete postfilter class provenance')
    if [r['name'] for r in rows]!=sorted(mapping.values()):
        raise ValueError('The accepted lexical class/gallery order changed')
    for r in rows:
        name=mapping[str(r['class_id'])]
        files=list((bank/'features'/name).glob('*.npz'))
        if len(files)!=1:raise ValueError('Missing/ambiguous class matrix')
        x,ids=load_feature_matrix_npz(files[0])
        expected_drop=0 if r['class_id']==0 else int(np.floor(r['N']*.05))
        if (r['name']!=name or r['dropped']!=expected_drop or r['kept']!=len(x)
                or ids.tolist()!=r['kept_ids'] or r['N']-expected_drop!=len(x)):
            raise ValueError('Filtered class members differ from provenance')
    if sum(r['kept'] for r in rows)!=report['total_kept']:
        raise ValueError('Postfilter total differs')
    return report


def gallery_class_order(bank, legacy_order):
    """Keep accepted post5 gallery rows; legacy banks retain their old order."""
    if (Path(bank)/'meta/postfilter.json').is_file():
        return [r['name'] for r in validate_filtered_bank(bank)['classes']]
    return list(legacy_order)
