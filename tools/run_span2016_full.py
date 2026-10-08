"""SPAN x4 native labels, two-GPU 2016 SMFA banks/eval, fresh three-layer MLP."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def parallel(jobs):
    """One independent subprocess per physical GPU; stop siblings on failure."""
    processes, streams = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Launcher interrupted; stopping workers')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for command, env, log in jobs:
            log.parent.mkdir(parents=True, exist_ok=True)
            write(log.with_suffix('.command.json'), dict(command=command, environment={k:v for k,v in env.items()
                if k.startswith('OVERRIDE_') or k == 'CUDA_VISIBLE_DEVICES'}))
            stream = log.open('w', encoding='utf-8'); streams.append(stream)
            processes.append(subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Worker failed; inspect stage worker logs')
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Worker failed; inspect stage worker logs')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        for stream in streams:
            stream.close()
        signal.signal(signal.SIGTERM, previous)


def check_coverage(shards, expected):
    actual = [i for shard in shards for i in shard]
    if len(actual) != len(set(actual)) or set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError('Shard coverage has missing/duplicate/unexpected IDs')


def merge_metrics(results, ids):
    import numpy as np
    check_coverage([r['evaluated_ids'] for r in results], ids)
    for key in ('feature_signature','feature_bank','cleanup','proposals_dir','inference_settings','feature_image_dir','feature_image_scale'):
        if any(r[key] != results[0][key] for r in results):
            raise ValueError(f'Evaluation shards differ in {key}')
    matrix = sum((np.asarray(r['confusion_matrix'], dtype=np.int64) for r in results), np.zeros((21,21),dtype=np.int64))
    union = matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.zeros(21,dtype=float), where=union>0)
    names = list(results[0]['per_class_iou'])
    if len(names) != 21 or any(list(r['per_class_iou']) != names for r in results):
        raise ValueError('Evaluation class order mismatch')
    return dict(results[0], mIoU=float(iou.mean()), per_class_iou=dict(zip(names,iou.tolist())),
        confusion_matrix=matrix.tolist(), evaluated_ids=ids, evaluated_count=len(ids), shard_index=None, shard_count=2,
        proposal_count_before=sum(r['proposal_count_before'] for r in results),
        proposal_count_after=sum(r['proposal_count_after'] for r in results), aggregation='sum_confusion_then_iou')


def build_worker(job):
    import torch
    from src.bank_input_validation import image_index
    from src.region_extractors import get_regions_from_pseudo
    from src.feature_extractor_factory import build_feature_extractor
    from src.build_feature_bank import create_dirs, save_class_mapping, FeatureBank
    from src.feature_signature import save_feature_signature
    from configs import config as cfg
    from project_utils.config_logger import record_config
    from project_utils.seed_utils import set_global_seed
    set_global_seed(cfg.SEED)
    create_dirs(); save_class_mapping(cfg.NUM_CLASSES,cfg.CLS_PATH)
    record_config(cfg.META_DIR,'run_build_feature_bank.py')
    model = build_feature_extractor(torch.device('cuda:0'),use_interactivity=False)
    import inspect
    trunk=model.model.backbone.vision_backbone.trunk
    source=Path(inspect.getfile(type(trunk)))
    from project_utils.experiment import sha256
    write(Path(cfg.META_DIR)/'runtime.json',dict(torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
        backbone_source=str(source),backbone_sha256=sha256(source),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    sig=model.feature_signature()
    if sig['resolution'] != 2016 or sig['feature_grid'] != [144,144] or sig.get('adapter'):
        raise ValueError('Wrong raw feature space')
    save_feature_signature(cfg.META_DIR,sig)
    images=image_index(cfg.BUILD_IMAGE_DIR); bank=FeatureBank(); counts={}
    for pos,stem in enumerate(job['ids'],1):
        regions=get_regions_from_pseudo(str(images[stem]),min_area=cfg.MIN_MASK_AREA)
        before=getattr(model,'_feature_failure_count',0)
        vectors=model.extract_features_for_regions(str(images[stem]),regions,0) if regions else {}
        if getattr(model,'_feature_failure_count',0)!=before or sum(map(len,vectors.values()))!=len(regions):
            raise RuntimeError(f'Incomplete feature extraction: {stem}')
        for cid, values in vectors.items():
            for value in values:
                if not bank._is_valid_feature(value):
                    raise ValueError(f'Invalid feature: {stem}')
                bank.add_feature('background' if cid==0 else cfg.DATASET_CLASSES[cid-1],value,stem)
        counts[stem]=len(regions)
        print(f'[Build] {pos}/{len(job["ids"])} {stem}: {len(regions)} regions',flush=True)
    bank.save_stats(); bank.build_indices()
    # Filter only after the two unfiltered banks have been merged.
    write(Path(cfg.META_DIR)/'coverage.json',dict(ids=job['ids'],region_counts=counts))


def merge_banks(job):
    import numpy as np
    from src.feature_bank_storage import load_feature_matrix_npz, save_feature_matrix_npz
    from src.filter_stage import run_filter_stage
    from project_utils.config_logger import record_config
    from configs import config as cfg
    shards=[Path(p) for p in job['shards']]; target=Path(cfg.OUTPUT_ROOT)
    check_coverage([read(s/'meta/coverage.json')['ids'] for s in shards],job['ids'])
    signature=read(shards[0]/'meta/feature_signature.json')
    mapping=read(shards[0]/'meta/class_mapping.json')
    for s in shards:
        if read(s/'meta/feature_signature.json')!=signature or read(s/'meta/class_mapping.json')!=mapping:
            raise ValueError('Raw shard signature/mapping mismatch')
    if target.exists():
        raise FileExistsError(target)
    for name in mapping['id_to_dirname'].values():
        matrices=[]; labels=[]
        for s in shards:
            for file in (s/'features'/name).glob('*.npz'):
                x, sample_ids=load_feature_matrix_npz(file)
                matrices.append(x); labels.extend(sample_ids.tolist())
        if len(set(labels))!=len(labels):
            raise ValueError(f'Duplicate bank sample IDs: {name}')
        if matrices:
            save_feature_matrix_npz(target/'features'/name/f'{name}_features.npz',np.concatenate(matrices),labels)
    (target/'index').mkdir(parents=True,exist_ok=True)
    write(target/'meta/feature_signature.json',signature); write(target/'meta/class_mapping.json',mapping)
    write(target/'meta/coverage.json',dict(ids=job['ids'],shards=job['shards']))
    record_config(str(target/'meta'),'run_build_feature_bank.py')
    run_filter_stage()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',default='spanx4_2016_noerosion_v1')
    p.add_argument('--train-images',type=Path,default=Path('/home/zystu2025/datasets/VOC2012/JPEGImages_SPAN_x4'))
    p.add_argument('--val-images',type=Path,default=Path('/home/zystu2025/datasets/VOC2012/JPEGImages'))
    p.add_argument('--val-image-scale',type=int,choices=[1,4],default=1,
                   help='Explicitly use original val images (1) or SPAN x4 (4); always encode at 2016')
    p.add_argument('--proposals-dir',type=Path,required=False,help='Existing raw validation proposal directory')
    p.add_argument('--imagelevel-json',type=Path,help='Training image-level labels; defaults to configured split')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/sam3/sam3.pt')
    p.add_argument('--split-file',type=Path,help='Optional MLP training_report.json to reuse image split')
    p.add_argument('--build-min-area',type=int,default=4800,help='300 original pixels x16')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',choices=['build','merge']); p.add_argument('--job',type=Path)
    args=p.parse_args(); os.chdir(ROOT)
    if args.worker:
        return (build_worker if args.worker=='build' else merge_banks)(read(args.job))
    if not args.val_images:
        p.error('--val-images is required: complete validation images')
    if not args.version or any(c in args.version for c in '/\\') or args.version in {'.','..'} or args.build_min_area<1:
        p.error('Invalid version/minimum area')
    if args.proposals_dir is None:
        p.error('--proposals-dir is required for the complete pipeline')
    from project_utils.experiment import stage, train_command, sha256, snapshot
    from configs.experiment import PSEUDO, TRAINING, CLEANUP, cli_options
    from configs import config as cfg
    from src.bank_input_validation import image_index, validate_pairs
    from PIL import Image
    import torch
    output=ROOT/'experiments'/args.version
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use unchanged --resume or a new --version')
    free=shutil.disk_usage(ROOT).free
    print(f'[Preflight] project disk free={free/2**30:.1f} GiB',flush=True)
    if free<2**30:
        raise OSError('Less than 1 GiB free on project filesystem; free space before starting')
    print('[Preflight] checking GPUs, fixed configuration and x4 train/val images',flush=True)
    for gpu in (0,1):
        value=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        mem,util=map(int,value.strip().split(','))
        if mem>1000 or util>10:
            raise RuntimeError(f'Physical GPU {gpu} occupied: {value.strip()}')
    recipe=dict(PSEUDO)
    if recipe['dataset']!='voc' or recipe['split'] not in ('train','trainaug'):
        raise ValueError('This two-GPU pipeline requires VOC train/trainaug')
    if TRAINING['num_layers']!=3:
        raise ValueError('This pipeline requires the configured three-layer MLP')
    dataset=ROOT/'data/VOC2012'; split=recipe['split']
    trainlist=dataset/f'ImageSets/Segmentation/{split}.txt'; vallist=dataset/'ImageSets/Segmentation/val.txt'
    trainids=trainlist.read_text().split(); valid=vallist.read_text().split()
    check_coverage([trainids],trainids); check_coverage([valid],valid)
    if not trainids or not valid:
        raise ValueError('Empty train/validation split')
    if set(trainids)&set(valid):
        raise ValueError('Training and final validation image lists overlap')
    proposals=args.proposals_dir.resolve(); ckpt=args.checkpoint.resolve()
    ckpthash=sha256(ckpt)
    labels_json=(args.imagelevel_json or dataset/f'ImageSets/ImageLevel/{split}_imagelevel.json').resolve()
    check_coverage([[Path(r['file_name']).stem for r in read(labels_json)['images']]],trainids)
    inputs=[ckpt,trainlist,vallist,labels_json]
    if args.split_file:
        inputs.append(args.split_file.resolve())
    # These effective values are explicit and saved, never restored from old experiments.
    build={'OVERRIDE_DATASET_TYPE':'voc','OVERRIDE_VOC_TRAIN_SPLIT':split}
    for key in ('SEED','BUILD_SPLIT_CONNECTED_COMPONENTS','FILTER_FOREGROUND_DROP_RATIO',
                'FILTER_BACKGROUND_DROP_RATIO','SAM3_FEATURE_SOURCE','SAM3_FEATURE_LEVEL',
                'SAM3_TOKEN_NORM','SAM3_CONTEXT_SUBTRACT','SAM3_USE_BF16'):
        value=getattr(cfg,key)
        build['OVERRIDE_'+key]=str(value).lower() if isinstance(value,bool) else str(value)
    inference={key:getattr(cfg,key) for key in (
        'INF_NMS_IOU','INF_TOPK_NEIGH','INF_RETRIEVAL_MODE','INF_CLASS_TOPM',
        'INF_RETRIEVAL_SEARCH_MULTIPLIER','INF_CENTROID_WEIGHT','INF_RETRIEVAL_MARGIN_THR',
        'INF_PROPOSAL_SCORE_WEIGHT','INF_INSTANCE_SCORE_THR','INF_FAISS_USE_GPU',
        'INF_FAISS_EXACT_SEARCH_THRESHOLD')}
    inference['MIN_MASK_AREA']=300
    for ids,folder,scale in ((trainids,args.train_images,4),(valid,args.val_images,args.val_image_scale)):
        index=image_index(folder)
        for pos,stem in enumerate(ids,1):
            source=dataset/f'JPEGImages/{stem}.jpg'
            if stem not in index:
                raise FileNotFoundError(f'Missing x4 image: {folder}/{stem}')
            with Image.open(source) as orig, Image.open(index[stem]) as high:
                if high.size!=(orig.width*scale,orig.height*scale):
                    raise ValueError(f'Not exact x{scale}: {index[stem]}')
            inputs += [source,index[stem],dataset/f'SegmentationClassAug/{stem}.png']
            if pos%500==0 or pos==len(ids):
                print(f'[Preflight] x{scale} images {pos}/{len(ids)}: {folder}',flush=True)
    inputs += [proposals/f'{stem}.npz' for stem in valid]
    sourcefiles=list(ROOT.glob('*.py'))
    for folder in ('src','tools','configs','project_utils'):
        sourcefiles+=list((ROOT/folder).rglob('*.py'))
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'},
                  inputs=snapshot(set(inputs)),code={str(f.relative_to(ROOT)):sha256(f) for f in sourcefiles},
                  recipe=recipe,train_args=TRAINING,build=build,inference=inference,
                  checkpoint_sha256=ckpthash,resolution=2016,query_chunk=256,
                  pseudo_encoder_resolution=1008,pseudo_output='native_x4',cleanup=CLEANUP)
    if output.exists():
        if not args.resume or read(output/'manifest.json')!=manifest:
            raise ValueError('Existing output: use unchanged --resume or a new version')
    else:
        write(output/'manifest.json',manifest)
    base={k:v for k,v in os.environ.items() if not k.startswith('OVERRIDE_')}
    base['PYTHONUNBUFFERED']='1'
    def launch(command,env,attempt,name='run'):
        parallel([(command,env,attempt/f'{name}.log')])
    def generate(attempt):
        dest=ROOT/'storage/pseudo_labels/voc'/args.version/attempt.name
        command=[sys.executable,'-u',str(ROOT/'tools/run_pseudo_two_gpus.py'),*cli_options(recipe),
            '--ckpt',str(ckpt),'--dataset-root',str(dataset),'--image-dir',str(args.train_images.resolve()),
            '--imagelevel-json',str(labels_json),'--output-dir',str(dest),'--experiment-dir',str(attempt/'generation'),
            '--eval-resize-prediction-to-gt','--evaluate']
        launch(command,dict(base,CUDA_VISIBLE_DEVICES='0,1'),attempt)
        validate_pairs(dest,args.train_images,trainids)
        return dict(pseudo=str(dest)),list(dest.rglob('*.png'))+list(dest.rglob('generation_*.json'))
    generated=stage(output,'pseudo',args.resume,generate)
    env=dict(base,**build)
    env.update(OVERRIDE_SAM3_CKPT=str(ckpt),OVERRIDE_SAM3_RESOLUTION='2016',
        OVERRIDE_FEATURE_ADAPTER_CKPT='',OVERRIDE_PSEUDO_MASK_ROOT=generated['pseudo'],
        OVERRIDE_PSEUDO_MASK_TYPE=args.version,OVERRIDE_BUILD_IMAGE_DIR=str(args.train_images.resolve()),
        OVERRIDE_MIN_MASK_AREA=str(args.build_min_area))
    def build_stage(attempt):
        shardroot=ROOT/'storage/raw_banks/voc'/args.version/attempt.name
        jobs=[]; shards=[]
        for gpu in (0,1):
            dest=shardroot/f'shard_{gpu}'; shards.append(str(dest))
            if dest.exists(): raise FileExistsError(dest)
            job=attempt/f'build_{gpu}.json'; write(job,dict(ids=trainids[gpu::2]))
            jobs.append(([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','build','--job',str(job)],
                         dict(env,CUDA_VISIBLE_DEVICES=str(gpu),OVERRIDE_OUTPUT_ROOT=str(dest)),attempt/f'build_{gpu}.log'))
        parallel(jobs)
        merged=shardroot/'merged'; job=attempt/'merge.json';write(job,dict(shards=shards,ids=trainids))
        launch([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','merge','--job',str(job)],
               dict(env,CUDA_VISIBLE_DEVICES='0',OVERRIDE_OUTPUT_ROOT=str(merged)),attempt,'merge')
        return dict(raw=str(merged)),list(merged.rglob('*.npz'))+list((merged/'meta').glob('*.json'))
    built=stage(output,'build',args.resume,build_stage)
    def train(attempt):
        weights=ROOT/'weights/sam3_adapter'/args.version/attempt.name
        bank=ROOT/'storage/mlp_banks/voc'/args.version/attempt.name
        command=train_command(built['raw'],weights,bank,trainlist,args.split_file,TRAINING)
        launch(command,dict(env,CUDA_VISIBLE_DEVICES='1'),attempt)
        from src.feature_adapter import adapted_signature
        trained_checkpoint=torch.load(weights/'adapter.pt',map_location='cpu',weights_only=True)
        source_signature=read(Path(built['raw'])/'meta/feature_signature.json')
        expected=adapted_signature(source_signature,weights/'adapter.pt',trained_checkpoint['architecture'])
        if trained_checkpoint['architecture'].get('num_layers')!=3 or read(bank/'meta/feature_signature.json')!=expected:
            raise ValueError('New adapter/exported bank signature mismatch')
        return dict(bank=str(bank),adapter=str(weights/'adapter.pt')),list(bank.rglob('*.npz'))+list((bank/'meta').glob('*.json'))+[weights/'adapter.pt',weights/'training_report.json']
    trained=stage(output,'train',args.resume,train)
    evalenv=dict(env)
    for key,value in inference.items():
        evalenv['OVERRIDE_'+key]=str(value).lower() if isinstance(value,bool) else str(value)
    evalenv.update(OVERRIDE_INF_FEATURE_BANK_ROOT=trained['bank'],OVERRIDE_FEATURE_ADAPTER_CKPT=trained['adapter'],
        OVERRIDE_INF_PROPOSALS_DIR=str(proposals),OVERRIDE_INF_LOAD_PROPOSALS='true',OVERRIDE_INF_VIS_LIMIT='0',
        OVERRIDE_INF_FAISS_DEVICE='0',OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK='false')
    cleanup=output/'cleanup.json';write(cleanup,CLEANUP)
    def evaluate(attempt):
        jobs=[]
        for gpu in (0,1):
            command=[sys.executable,'-u',str(ROOT/'run_inference.py'),'--strict-eval','--cleanup-config',str(cleanup),
                '--feature-image-dir',str(args.val_images.resolve()),'--shard-count','2','--shard-index',str(gpu),
                '--feature-image-scale',str(args.val_image_scale),
                '--results-json',str(attempt/f'metrics_{gpu}.json'),'--output-dir',str(attempt/f'eval_{gpu}')]
            jobs.append((command,dict(evalenv,CUDA_VISIBLE_DEVICES=str(gpu)),attempt/f'eval_{gpu}.log'))
        parallel(jobs)
        result=merge_metrics([read(attempt/f'metrics_{i}.json') for i in (0,1)],valid)
        if result['feature_signature']!=read(Path(trained['bank'])/'meta/feature_signature.json') or result['cleanup']!=CLEANUP:
            raise ValueError('Final evaluation signature/cleanup differs from experiment')
        write(attempt/'metrics.json',result)
        return dict(metrics=str(attempt/'metrics.json')),list(attempt.glob('metrics*.json'))
    evaluated=stage(output,'eval',args.resume,evaluate)
    if snapshot(set(inputs))!=manifest['inputs']:
        raise RuntimeError('Input files changed during experiment')
    write(output/'result.json',read(evaluated['metrics']))
    print(f'[DONE] mIoU={read(evaluated["metrics"])["mIoU"]*100:.4f}% ; {output / "result.json"}',flush=True)


if __name__=='__main__':
    main()
