"""Activate a completed, verified presence1/edge/geometry MLP2/HCR experiment."""
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def checked_recipe(manifest, generation):
    from configs.experiment import PSEUDO_CLEANUP
    from src.pseudo_label_cleanup import POLICY
    if manifest.get('source_version'):
        if manifest.get('cleanup_policy')!=POLICY:
            raise ValueError('Cached experiment has no mainline semantic cleanup')
        geometry=manifest.get('label_cleanup',{})
    else:
        if not manifest.get('semantic_cleanup'):
            raise ValueError('Fresh experiment has no mainline semantic cleanup')
        geometry=manifest.get('label_cleanup',{})
    if any(geometry.get(k)!=v for k,v in PSEUDO_CLEANUP.items()):
        raise ValueError('Semantic cleanup geometry differs from the mainline')
    defaults=generation['pseudo'];extra=generation.get('pseudo_extra_cli',[])
    def option(name,default,cast=float):
        flag='--'+name.replace('_','-');value=default
        for index,arg in enumerate(extra):
            if arg==flag:value=cast(extra[index+1])
        return value
    def enabled(name):
        value=defaults.get(name,False);flag='--'+name.replace('_','-')
        for arg in extra:
            if arg==flag:value=True
            elif arg=='--no-'+name.replace('_','-'):value=False
        return value
    if not enabled('edge_repair') or not enabled('force_presence_one'):
        raise ValueError('Generation must enable edge repair and known-class presence=1')
    expected=dict(mask_logit_threshold=.4,close_kernel=3,close_iterations=1,dilate_pixels=1,diningtable_close_iterations=2)
    if any(option(k,defaults.get(k,v))!=v for k,v in expected.items()):
        raise ValueError('Edge repair differs from the mainline')


def activate(root,version):
    from configs.experiment import MAINLINE
    from tools.run_mainline_full import read,write,identity,verify,files_under
    from src.hard_center_gmp import describe
    root=Path(root).resolve();folder=root/'experiments'/version
    summary=read(folder/'summary.json');manifest=read(folder/'manifest.json')
    if (summary['status']!='complete' or summary['evaluated_count']!=1449 or summary['train_count']!=10582
            or summary['seed']!=42 or read(folder/'status.json')['status']!='complete'):
        raise ValueError('Only complete seed42, 10582-train/1449-val experiments can be activated')
    generation=read(root/'experiments'/manifest['source_version']/'manifest.json') if manifest.get('source_version') else manifest
    checked_recipe(manifest,generation)
    training=read(Path(summary['weights'])/'training_report.json')
    if len(training['epochs'])!=50 or training['args']['num_layers']!=2 or training['args']['recipe']!='v1':
        raise ValueError('Expected complete 50-epoch MLP2/v1 training')
    bank=Path(summary['mlp_bank']);adapter=Path(summary['weights'])/'adapter.pt'
    signature=read(bank/'meta/feature_signature.json');checkpoint=Path(manifest['checkpoint'])
    architecture=signature.get('adapter',{}).get('architecture',{})
    if (signature.get('aggregation')!=describe(.1) or signature.get('embed_dim')!=256
            or tuple(architecture.get(k) for k in ['input_dim','hidden_dim','output_dim'])!=(1280,512,256)
            or architecture.get('num_layers',2)!=2 or signature.get('use_bf16') is not False):
        raise ValueError('Selected bank signature is not the frozen FP32 GMP2 mainline')
    old_records=dict(manifest.get('reused_artifacts',{}))
    for phase in ['build','merge','train','eval']:
        marker=read(folder/phase/'complete.json')
        old_records.update({r['path']:r for r in marker['artifacts']})
    if not manifest.get('reused_artifacts'):
        for phase in ['proposals','cleanup']:
            old_records.update({r['path']:r for r in read(folder/phase/'complete.json')['artifacts']})
    old_records.update({r['path']:r for r in manifest['inputs']})
    paths=dict(bank=bank,adapter=adapter,checkpoint=checkpoint,dataset_root=Path(manifest['dataset']),
        clean_proposals_dir=Path(summary['clean_proposals']),raw_proposals_dir=Path(summary['raw_proposals']))
    targets=files_under([bank])+[adapter,checkpoint,paths['clean_proposals_dir']/'cleanup_config.json']
    targets+=list(paths['clean_proposals_dir'].glob('*.npz'))+list(paths['raw_proposals_dir'].glob('*.npz'))
    targets += [paths['dataset_root']/'ImageSets/Segmentation/val.txt']
    records=[]
    for target in sorted(set(targets)):
        key=str(target.resolve())
        if key not in old_records:raise ValueError(f'Unregistered selected input: {target}')
        records.append(old_records[key])
    verify(records)
    if signature['adapter']['sha256']!=identity(adapter)['sha256'] or signature['checkpoint_sha256']!=identity(checkpoint)['sha256']:
        raise ValueError('Selected bank, adapter and base checkpoint hashes differ')
    controls=[folder/'summary.json',folder/'metrics.json',folder/'manifest.json',folder/'status.json']
    record=dict(format_version=1,status='active',recipe_id=MAINLINE['recipe_id'],source_version=version,
        seed=42,mIoU=summary['mIoU'],best_epoch=summary['best_epoch'],evaluated_count=1449,
        activated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        paths={k:str(v.relative_to(root)) if v.is_relative_to(root) else str(v) for k,v in paths.items()},
        artifact_bindings=records+[identity(f) for f in controls])
    archived=root/'experiments/mainline_selections'/(version+'.json')
    if archived.exists():
        previous=read(archived);verify(previous['artifact_bindings'])
        if previous['paths']!=record['paths'] or previous['recipe_id']!=record['recipe_id']:
            raise ValueError('Historical selection changed; choose a new activation version')
        record=previous
    else:write(archived,record)
    pointer=root/'experiments/mainline_active.json'
    write(pointer,record)
    print(f'[Activated] {version}: mIoU={record["mIoU"]*100:.4f}%, best epoch={record["best_epoch"]}; {pointer}',flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--version',required=True)
    a=p.parse_args()
    if not all(c.isalnum() or c in '_-' for c in a.version):p.error('Provide a safe completed experiment version')
    activate(a.root,a.version)


if __name__=='__main__':main()
