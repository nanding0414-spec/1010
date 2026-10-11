"""Resolve an explicit, locally activated mainline without altering its experiment."""
from pathlib import Path

INPUT_NAMES = ('bank','adapter','checkpoint','dataset_root','clean_proposals_dir','raw_proposals_dir')


def resolve_mainline_inputs(args, root):
    from configs.experiment import MAINLINE
    from tools.run_mainline_full import read, verify
    root=Path(root)
    needed=any(getattr(args,name,None) is None for name in INPUT_NAMES)
    selection=None
    if needed and args.mainline_manifest.is_file():
        selection=read(args.mainline_manifest)
        if selection.get('recipe_id')!=MAINLINE['recipe_id'] or selection.get('status')!='active':
            raise ValueError('Selected mainline recipe is missing or incompatible')
        verify(selection['artifact_bindings'])
        if args.seed is not None and args.seed!=selection['seed']:
            raise ValueError('Requested seed differs from the selected mainline weights')
        for name in INPUT_NAMES:
            if getattr(args,name,None) is None:
                setattr(args,name,(root/selection['paths'][name]).resolve())
        args.seed=selection['seed']
    if args.checkpoint is None:
        args.checkpoint=root/'weights/sam3/sam3.pt'
    missing=[name for name in INPUT_NAMES if getattr(args,name,None) is None]
    if missing:
        raise ValueError('Provide explicit inputs or an activated --mainline-manifest; missing: '+', '.join(missing))
    args.seed=42 if args.seed is None else args.seed
    return selection
