"""Convert accepted balanced LSA artifacts to the portable mainline format.

No training, model selection, SAM3, proposals or evaluation is performed.
The source checkpoint/export remain unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checked(record):
    p = Path(record['path']);h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):h.update(block)
    if p.stat().st_size != record['size'] or h.hexdigest() != record['sha256']:
        raise ValueError(f'Source bytes differ: {p}')
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-acceptance', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[42,123,2026], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from src.feature_adapter import FeatureAdapter, adapted_signature, project_features
    from src.feature_bank_storage import save_feature_matrix_npz
    from src.semantic_bank_filter import export_filtered_bank
    m=json.loads(a.source_manifest.read_text());acc=json.loads(a.source_acceptance.read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!='binding'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (m['binding'] != digest or acc['binding'] != digest or acc['status'] != 'passed'
            or acc['manifest_SHA'] != hashlib.sha256(a.source_manifest.read_bytes()).hexdigest()
            or m['version'] != 'small_mask_hcr_20261027_200000'):
        p.error('Require the independently accepted area100 source manifest and acceptance')
    row=m['B_sources'][str(a.seed)];report=json.loads(checked(row['training_report']).read_text())
    if report['best_checkpoint'] != row['checkpoint'] or report['best_epoch'] != row['best_epoch']:
        p.error('Selected balanced checkpoint differs from its signed training report')
    payload=torch.load(checked(row['checkpoint']),map_location='cpu',weights_only=False)
    if payload['format_version'] != 'bank_filter_placement_v1' or payload['arm'] != 'B_post10':
        p.error('Unexpected source checkpoint architecture/recipe')
    if payload['seed'] != a.seed or payload['epoch'] != report['best_epoch'] or payload['source_signature'] != m['original_signature']:
        p.error('Checkpoint seed/epoch/raw signature differs')
    with np.load(checked(row['export']),allow_pickle=False) as z:
        vectors,ids,labels,classes=[z[k].copy() for k in ['z','ids','y','classes']]
    model=FeatureAdapter(1280,512,256,num_layers=2);model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    with np.load(checked(m['prepared_zero']),allow_pickle=False) as z:raw=z['x'][:1024].copy()
    actual=project_features(model,raw,torch.device('cpu'))
    np.testing.assert_allclose(actual,vectors[:1024],atol=1e-6,rtol=1e-6)
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    mapping=json.loads((Path(m['native_raw_bank'])/'meta/class_mapping.json').read_text())
    if set(classes.tolist()) != set(mapping['id_to_dirname'].values()):raise ValueError('Class mapping differs')
    a.output_dir.mkdir(parents=True)
    ckpt=a.output_dir/'adapter.pt'
    torch.save(dict(format_version=1,architecture=model.architecture,state_dict=payload['model'],
        source_signature=m['original_signature'],classes=classes.tolist(),epoch=payload['epoch'],
        validation=payload.get('metrics',{}),recipe_id='lsa_hcgmp_prr_v1',
        source_checkpoint=row['checkpoint']),ckpt)
    bank=a.output_dir/'unfiltered_bank';(bank/'meta').mkdir(parents=True)
    (bank/'meta/class_mapping.json').write_text(json.dumps(mapping,indent=2))
    (bank/'meta/feature_signature.json').write_text(json.dumps(adapted_signature(m['original_signature'],ckpt,model.architecture),indent=2))
    for i,name in enumerate(classes.tolist()):
        loc=np.flatnonzero(labels==i);save_feature_matrix_npz(bank/'features'/name/(name+'_features.npz'),vectors[loc],ids[loc])
    filtered=export_filtered_bank(bank,a.output_dir/'bank')
    receipt=dict(recipe_id='lsa_hcgmp_prr_v1',seed=a.seed,selected_epoch=payload['epoch'],
        source_binding=m['binding'],source_checkpoint=row['checkpoint'],source_export=row['export'],
        maxabs1024=float(np.max(np.abs(actual-vectors[:1024]))),training_steps=0,
        postfilter_total_kept=filtered['total_kept'],GPU_quality_run=False)
    (a.output_dir/'export_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
