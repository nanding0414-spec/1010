#!/usr/bin/env python3
"""Verified, non-destructive import and regression of the original-image record.

Historical paths are relocated in memory only. Native inference does not install
experimental aggregation, adapter or signature patches. Historical feature caches
are used solely by this deployment validator; run_mainline.py extracts online.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE/'sam3-main')]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relocate(value, root):
    old='/home/zystu2025/nd/ModuSeg_SAM3'
    if isinstance(value,str):
        return str(root)+value[len(old):] if value==old or value.startswith(old+'/') else value
    if isinstance(value,list):return [relocate(x,root) for x in value]
    if isinstance(value,dict):return {relocate(k,root):relocate(v,root) for k,v in value.items()}
    return value


def read(path, root):
    return relocate(json.loads(Path(path).read_text(encoding='utf-8')),root)


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def checked_source(job):
    from tools.experimental_mpsa_combo_queue import Recipe, signature
    source=job['raw_signature'];base={k:v for k,v in source.items() if k not in ('experimental_pooling','format_version')}
    base['format_version']=2
    if source!=signature(base,'combo',Recipe('C',job['checkpoint_sha256'])):
        raise ValueError('Only the verified original-image single-phase uniform GMP rho=.1 source is importable')
    if digest(job['checkpoint'])!=job['checkpoint_sha256']:
        raise ValueError('SAM3 checkpoint changed')
    from src.hard_center_gmp import describe
    return dict(base,format_version=3,aggregation=describe(),checkpoint_sha256=job['checkpoint_sha256'],
                precision_policy='standard_vit_mlp_fp32_v1')


def import_model(job, root, version, native):
    import torch
    from src.feature_adapter import FeatureAdapter, load_adapter, adapted_signature
    old=Path(job['weights'])/'adapter.pt';bank=Path(job['bank'])
    payload=torch.load(old,map_location='cpu',weights_only=True)
    from tools.experimental_smfa_mlp_queue import adapter_tag
    architecture=payload['architecture']
    if (payload['source_signature']!=job['raw_signature'] or architecture!=dict(input_dim=1280,
        hidden_dim=512,output_dim=256,num_layers=2,experimental_mlp=adapter_tag('d2'))):
        raise ValueError('Wrong source signature or MLP architecture')
    original_sig=read(bank/'meta/feature_signature.json',root)
    if original_sig.get('adapter',{}).get('sha256')!=digest(old):
        raise ValueError('Historical bank adapter hash mismatch')
    if {k:v for k,v in original_sig.items() if k not in ('adapter','embed_dim')}!={k:v for k,v in job['raw_signature'].items() if k!='embed_dim'}:
        raise ValueError('Historical projected-bank feature space differs')
    target=root/'weights/sam3_adapter'/version/str(job['seed'])
    output=root/'storage/mlp_banks/voc'/version/str(job['seed'])
    if target.exists() or output.exists():raise FileExistsError('New import destinations must be unused')
    target.mkdir(parents=True)
    model=FeatureAdapter(1280,512,256,2);model.load_state_dict(payload['state_dict'],strict=True)
    converted=copy.deepcopy(payload);converted['architecture']=model.architecture
    converted['source_signature']=native
    converted['mainline_import']=dict(source_checkpoint_sha256=digest(old),source_bank=str(bank),
        source_signature=job['raw_signature'],no_retraining=True,unchanged_state_dict=True)
    torch.save(converted,target/'adapter.pt')
    loaded,_=load_adapter(target/'adapter.pt',source_signature=native)
    torch.manual_seed(42);x=torch.randn(257,1280)
    with torch.inference_mode():
        if not torch.equal(model(x),loaded(x)):raise ValueError('Imported MLP changed its output')
    shutil.copytree(bank,output)
    write(output/'meta/feature_signature.json',adapted_signature(native,target/'adapter.pt',model.architecture))
    write(output/'meta/mainline_import.json',converted['mainline_import'])
    # Rewrite provenance only in the new copy; feature/index files remain byte-identical.
    export=output/'meta/adapter_export.json'
    if export.exists():
        d=read(export,root);d.update(adapter_checkpoint=str(target/'adapter.pt'),source_signature=native)
        if 'architecture' in d:d['architecture']=model.architecture
        write(export,d)
    for file in bank.rglob('*'):
        if file.is_file() and file.relative_to(bank).parts[0] in ('features','index'):
            if digest(file)!=digest(output/file.relative_to(bank)):raise ValueError('Bank arrays/index changed during import')
    return dict(adapter=str(target/'adapter.pt'),bank=str(output),adapter_sha256=digest(target/'adapter.pt'))


def import_raw(job, root, version, native):
    import torch
    from src.feature_bank_storage import load_feature_matrix_npz
    payload=torch.load(Path(job['weights'])/'adapter.pt',map_location='cpu',weights_only=True)
    source=Path(relocate(payload['train_args']['source_bank'],root))
    if read(source/'meta/feature_signature.json',root)!=job['raw_signature']:
        raise ValueError('Raw source bank signature differs from the verified model')
    count=0
    projected=Path(job['bank'])
    for file in sorted((source/'features').glob('*/*.npz')):
        vectors,ids=load_feature_matrix_npz(file)
        _,projected_ids=load_feature_matrix_npz(projected/file.relative_to(source))
        if ids.tolist()!=projected_ids.tolist():raise ValueError('Raw/projected members differ')
        count+=len(vectors)
    if count!=37200:raise ValueError(f'Expected 37200 original regions, got {count}')
    target=root/'storage/raw_banks/voc'/version
    if target.exists():raise FileExistsError(target)
    shutil.copytree(source,target)
    write(target/'meta/feature_signature.json',native)
    write(target/'meta/mainline_import.json',dict(source=str(source),source_signature=job['raw_signature'],
        members=count,no_refiltering=True,unchanged_vectors=True))
    for file in (source/'features').rglob('*.npz'):
        if digest(file)!=digest(target/file.relative_to(source)):raise ValueError('Raw vectors changed during copy')
    return str(target)


def cached_worker(spec):
    import numpy as np
    import torch
    from PIL import Image
    from tools import experimental_pmfa_ab as ab
    from tools import experimental_mpsa_combo_queue as queue
    from tools import experimental_overnight_rescue as prior
    from tools.experimental_gmp_seeds import image_confusions
    root=Path(spec['root']);job=spec['job'];native=spec['native'];state={}
    ab.read=lambda p:read(p,root)  # Path relocation only; signatures are still strict/native.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from src.feature_adapter import load_adapter, adapted_signature, project_features
    class Cached:
        def __init__(self,device):
            self.device=device;self.adapter,_=load_adapter(spec['model']['adapter'],device,native);self.embed_dim=256
        def feature_signature(self):return adapted_signature(native,spec['model']['adapter'],self.adapter.architecture)
        def extract_features_batch(self,rgb,masks):
            stem=state['stem']
            if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()!=state['rgb_hash']:
                raise ValueError('Query RGB changed')
            hashes=queue.mask_hashes(masks)
            strong,_,extras,_,_=prior.pool(job,stem)
            strong_hashes=queue.mask_hashes(strong);extra_hashes=queue.mask_hashes(extras)
            wanted=strong_hashes+extra_hashes if spec['hcr'] else strong_hashes
            if hashes!=wanted:raise ValueError('Candidate mask content/order changed')
            raw=queue.cache_vectors(Path(job['cache'])/'val'/stem,job['cache_identity'][stem],'C',strong_hashes)
            if spec['hcr']:
                added,stored=prior.cached_extra(job,stem)
                if stored!=extra_hashes:raise ValueError('HCR cache mask order changed')
                raw=np.concatenate([raw,added],axis=0)
            return list(project_features(self.adapter,raw,self.device))
    import project_utils.metrics as metrics
    original_eval=metrics.evaluate_dataset_miou
    def evaluate(preds,gts,count,ignore,**kwargs):
        if len(preds)!=1449:raise ValueError('Full 1449-image validation required')
        matrices=image_confusions(preds,gts,count,ignore)
        np.savez_compressed(Path(spec['output'])/'per_image_confusion.npz',ids=np.asarray(job['val_ids']),confusions=matrices)
        return original_eval(preds,gts,count,ignore,**kwargs)
    metrics.evaluate_dataset_miou=evaluate
    import configs.config as cfg
    cfg.INF_DATASET_ROOT=job['dataset_root']
    import run_inference as inference
    inference.build_feature_extractor=lambda device,**_:Cached(device)
    original_load=inference.load_strict_proposals
    def load(directory,stem,h,w):
        identity=job['cache_identity'][stem]
        if digest(job['images'][stem])!=identity['image_sha256'] or digest(Path(directory)/(stem+'.npz'))!=identity['mask_source_sha256']:
            raise ValueError('Query input changed')
        state['stem']=stem
        with Image.open(job['images'][stem]) as im:state['rgb_hash']=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
        return original_load(directory,stem,h,w)
    inference.load_strict_proposals=load
    args=['--strict-eval','--output-dir',spec['output'],'--results-json',str(Path(spec['output'])/'metrics.json')]
    if spec['hcr']:args+=['--rescue','--rescue-raw-dir',job['raw_proposals']]
    inference.main(args)
    actual=read(Path(spec['output'])/'metrics.json',root);expected=read(spec['reference'],root)
    if actual['confusion_matrix']!=expected['confusion_matrix']:raise ValueError('Full confusion matrix differs from history')
    with np.load(Path(spec['output'])/'per_image_confusion.npz') as a,np.load(Path(spec['reference']).parent/'per_image_confusion.npz') as b:
        if not np.array_equal(a['ids'],b['ids']) or not np.array_equal(a['confusions'],b['confusions']):
            raise ValueError('Per-image matrices differ from history')
    write(Path(spec['output'])/'complete.json',dict(identical_to_history=True,metrics=str(Path(spec['output'])/'metrics.json'),
        metrics_sha256=digest(Path(spec['output'])/'metrics.json'),per_image_sha256=digest(Path(spec['output'])/'per_image_confusion.npz')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=CODE)
    p.add_argument('--history',default='hcr_mainline_20261004_214541')
    p.add_argument('--version',required=False)
    p.add_argument('--worker',type=Path)
    a=p.parse_args();os.chdir(CODE)
    if a.worker:return cached_worker(json.loads(a.worker.read_text()))
    if not a.version or Path(a.version).name!=a.version:p.error('--version must be a single new directory name')
    root=a.root.resolve();folder=root/'experiments'/a.version
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    history=root/'experiments'/a.history
    if read(history/'summary.json',root)['status']!='complete':raise ValueError('Historical deployment is incomplete')
    results={};models={};specs=[]
    for seed in (42,123,2026):
        reference=history/f'validation/high95_only_{seed}/attempt_001'
        job=read(reference/'job.json',root);native=checked_source(job)
        if seed==42:raw_bank=import_raw(job,root,a.version,native)
        model=import_model(job,root,a.version,native);models[str(seed)]=model
        for mode in ('baseline','high95_only'):
            source=history/f'validation/{mode}_{seed}/attempt_001'
            spec=dict(root=str(root),job=job,native=native,model=model,hcr=mode!='baseline',
                reference=str(source/'metrics.json'),output=str(folder/f'{mode}_{seed}'))
            specs.append((spec,read(source/'command.json',root)['environment']))
    write(folder/'models.json',models)
    write(folder/'raw_bank.json',dict(path=raw_bank))
    # Native SAM3 probe before any cache regression; compare actual extraction to old vectors.
    import numpy as np
    import torch
    from PIL import Image
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    from tools import experimental_pmfa_ab as ab
    from tools import experimental_mpsa_combo_queue as queue
    ab.read=lambda path:read(path,root)
    job=specs[0][0]['job'];model=SAM3FeatureExtractor(model_path=job['checkpoint'],device=torch.device('cuda:0'),use_bf16=False,use_interactivity=False)
    if model.feature_signature()!=specs[0][0]['native']:raise ValueError('Native source signature differs')
    stem=job['val_ids'][0]
    with Image.open(job['images'][stem]) as im:rgb=np.asarray(im.convert('RGB'))
    with np.load(Path(job['proposals'])/(stem+'.npz')) as d:masks=list(d['masks'].astype(bool))
    with torch.inference_mode():live=np.asarray(model.extract_features_batch(rgb,masks))
    old=queue.cache_vectors(Path(job['cache'])/'val'/stem,job['cache_identity'][stem],'C',queue.mask_hashes(masks))
    if not np.allclose(live,old,atol=5e-6,rtol=5e-6):raise ValueError('Native GMP probe differs from historical vectors')
    write(folder/'probe.json',dict(image=stem,regions=len(masks),maximum_absolute_error=float(np.max(np.abs(live-old))),
        signature=model.feature_signature(),peak_bytes=torch.cuda.max_memory_allocated()))
    del model;torch.cuda.empty_cache()
    from concurrent.futures import ThreadPoolExecutor
    def lane(gpu,tasks):
        for spec,env in tasks:
            key=Path(spec['output']).name;spec_file=folder/(key+'_job.json');write(spec_file,spec)
            run_env=os.environ.copy();run_env.update(env)
            run_env.update(CUDA_VISIBLE_DEVICES=str(gpu),OVERRIDE_INF_FEATURE_BANK_ROOT=spec['model']['bank'],
                OVERRIDE_FEATURE_ADAPTER_CKPT=spec['model']['adapter'],OVERRIDE_SAM3_POOLING='hard_center_gmp',
                PYTHONDONTWRITEBYTECODE='1')
            log=root/'logs'/a.version/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as f:subprocess.run([sys.executable,'-B',__file__,'--worker',str(spec_file)],env=run_env,cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('[Verified]',key,flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,gpu,specs[gpu::2]) for gpu in (0,1)]
            for f in futures:f.result()
        for spec,_ in specs:
            metric=read(Path(spec['output'])/'metrics.json',root)
            cm=np.asarray(metric['confusion_matrix'],dtype=np.float64);iou=np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))
            results[Path(spec['output']).name]=dict(mIoU=float(iou.mean()),identical_to_history=True)
        write(folder/'summary.json',dict(status='complete',results=results,
            mean_hcr=float(np.mean([v['mIoU'] for k,v in results.items() if k.startswith('high95')])),
            note='Native projection/retrieval/HCR; historical pre-MLP vectors; one real SAM3/GMP query probe.'))
    except BaseException as exc:
        write(folder/'status.json',dict(status='failed',error=str(exc)));raise
    write(folder/'status.json',dict(status='complete'))


if __name__=='__main__':main()
