"""Opt-in square GMP experiments, isolated from the native feature protocol."""
import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def description(resolutions=(1008,),rho=.1,weighting='uniform'):
    resolutions=tuple(map(int,resolutions))
    if (not resolutions or len(set(resolutions))!=len(resolutions) or any(r not in (672,1008,1344) for r in resolutions)
            or weighting not in ('uniform','spatial_core') or not math.isfinite(rho) or rho<=0):
        raise ValueError('Invalid experimental GMP recipe')
    return dict(type='hard_center_gmp',version=3,protocol='square_gmp_tuning_v1',resolutions=list(resolutions),
        rho=float(rho),weights=weighting,geometry='square_center_floor_original_nearest_centroid',
        scale_fusion='mean_of_l2_vectors_then_l2',solver='fp32_cholesky_smaller_primal_or_dual',residual_limit=1e-4,
        spatial_weights='0.5_plus_distance_over_max' if weighting=='spatial_core' else None)

def validate_signature(s):
    a=s.get('aggregation',{})
    try:expected=description(a['resolutions'],a['rho'],a['weights'])
    except (KeyError,TypeError,ValueError):return False
    return (a==expected and s.get('resolution')==1008 and s.get('feature_grid')==[72,72] and s.get('embed_dim')==1280
        and s.get('feature_source')=='trunk_fpn' and s.get('token_norm')=='layernorm' and s.get('use_bf16') is False
        and s.get('context_subtract')==0 and len(s.get('scale_encoders',[]))==len(a['resolutions'])
        and all(e.get('resolution')==r and e.get('feature_grid')==[r//14]*2 and e.get('embed_dim')==1280
            and e.get('use_bf16') is False and e.get('checkpoint_sha256')==s.get('checkpoint_sha256')
            and e.get('feature_source')=='trunk_fpn' and e.get('feature_level')==-1
            and e.get('token_norm')=='layernorm' and e.get('context_subtract')==0
            and e.get('precision_policy')=='standard_vit_mlp_fp32_v1' and not e.get('adapter')
            for r,e in zip(a['resolutions'],s['scale_encoders'])))

def weighted_gmp(tokens,indices,weights,rho=.1):
    from src.hard_center_gmp import uniform_gmp
    if weights is None:return uniform_gmp(tokens,indices,rho)
    if tokens.ndim!=2 or not tokens.is_floating_point() or not torch.isfinite(tokens).all():raise ValueError('Invalid tokens')
    if not math.isfinite(rho) or rho<=0:raise ValueError('Invalid ridge ratio')
    f=tokens[torch.as_tensor(indices,dtype=torch.long,device=tokens.device)].float()
    p=torch.as_tensor(weights,dtype=torch.float32,device=f.device)
    if p.shape!=(len(f),) or not torch.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid spatial weights')
    if not len(f):return tokens.new_zeros(tokens.shape[-1],dtype=torch.float32)
    p=p/p.sum();energy=(p*f.square().sum(1)).sum();ridge=rho*energy
    if float(energy)==0:return tokens.new_zeros(tokens.shape[-1],dtype=torch.float32)
    a=p.sqrt()[:,None]*f;mu=p@f;primal=f.shape[1]<=len(f)
    matrix=a.T@a if primal else a@a.T;matrix.diagonal().add_(ridge)
    chol,flag=torch.linalg.cholesky_ex(matrix)
    if int(flag)!=0:raise ValueError('Weighted GMP Cholesky failed')
    x=torch.cholesky_solve((mu if primal else p.sqrt())[:,None],chol).squeeze(1);z=x if primal else a.T@x
    residual=(f.T@(p*(f@z))+ridge*z-mu).norm()/mu.norm().clamp_min(1e-12)
    if not torch.isfinite(z).all() or float(residual)>1e-4:raise ValueError('Weighted GMP residual exceeds limit')
    return F.normalize(z,dim=0)

def spatial_weights(mask,grid,indices):
    import cv2
    # Mask-interior distance is a spatial proxy, not Transformer attention.
    distance=cv2.distanceTransform(np.asarray(mask,np.uint8),cv2.DIST_L2,5);gh,gw=grid
    yy,xx=np.meshgrid((np.arange(gh)+.5)*mask.shape[0]/gh,(np.arange(gw)+.5)*mask.shape[1]/gw,indexing='ij')
    values=distance[yy.astype(int).clip(0,mask.shape[0]-1),xx.astype(int).clip(0,mask.shape[1]-1)].ravel()[indices]
    return .5+values/max(float(values.max()) if len(values) else 0.,1e-6)

class TuningExtractor:
    def __init__(self,checkpoint,device,folders,source_signature,resolutions=(1008,),rho=.1,weighting='uniform',read_only=False):
        from src.feature_extractor_sam3 import SAM3FeatureExtractor
        from src.image_token_cache import ImageTokenCache
        self.device=torch.device(device);self.embed_dim=1280;self.models=[];self.recipe=description(resolutions,rho,weighting)
        for r in resolutions:
            folder=Path(folders[str(r)]);receipt=next(folder.glob('*.json'),None) if folder.is_dir() else None
            if r==1008:encoder={k:v for k,v in source_signature.items() if k not in ('aggregation','adapter')}
            elif receipt:
                import json
                encoder=json.loads(receipt.read_text())['binding']['encoder_signature']
            else:encoder=None
            if read_only or r==1008:
                if encoder is None:raise FileNotFoundError(f'No scale encoder receipt: {folder}')
                model=SAM3FeatureExtractor.__new__(SAM3FeatureExtractor);model.device=self.device;model.embed_dim=1280;readonly=True
            else:
                model=SAM3FeatureExtractor(model_path=checkpoint,device=self.device,resolution=r,pooling='smfa',use_bf16=False,
                    enable_segmentation=False,use_interactivity=False,feature_source='trunk_fpn',feature_level=-1,
                    token_norm='layernorm',context_subtract=0)
                encoder=model.feature_signature();readonly=False
            if (encoder.get('resolution')!=r or encoder.get('feature_grid')!=[r//14]*2
                    or encoder.get('checkpoint_sha256')!=source_signature['checkpoint_sha256']):raise ValueError('Scale encoder differs')
            model.image_token_cache=ImageTokenCache(folder,encoder,read_only=readonly);self.models.append((r,model,encoder))
        self._signature=copy.deepcopy(source_signature);self._signature['aggregation']=self.recipe
        self._signature['scale_encoders']=[e for _,_,e in self.models]
        if not validate_signature(self._signature):raise ValueError('Invalid scale signature')
    def feature_signature(self):return copy.deepcopy(self._signature)
    def set_image_source(self,path,rgb):
        self.source=Path(path)
        for _,model,_ in self.models:model.image_token_cache.bind(path,rgb)
    @torch.no_grad()
    def extract_features_batch(self,rgb,masks):
        from src.geometry_gmp import select_centers
        scales=[]
        for r,model,_ in self.models:
            tokens,grid=model.image_token_cache.tokens(model,rgb);vectors=[]
            for mask in masks:
                if np.asarray(mask).shape!=rgb.shape[:2]:raise ValueError('Mask/image shape differs')
                indices=select_centers(mask,r,grid,letterbox=False)
                weights=spatial_weights(mask,grid,indices) if self.recipe['weights']=='spatial_core' else None
                vectors.append(weighted_gmp(tokens[0],indices,weights,self.recipe['rho']))
            scales.append(torch.stack(vectors) if vectors else tokens.new_zeros((0,1280)))
        result=F.normalize(torch.stack(scales).mean(0),dim=1)
        if not torch.isfinite(result).all():raise ValueError('Invalid fused feature')
        return list(result.cpu().numpy())
    def extract_features(self,rgb,mask):return torch.as_tensor(self.extract_features_batch(rgb,[mask])[0],device=self.device)
    def _warn_feature_failure(self,error):raise RuntimeError('Strict experimental feature failure') from error

class QueryCache:
    """Pre-MLP queries bound to source, mask list, signature and encoder receipts."""
    def __init__(self,model,folder):
        self.base=model;self.folder=Path(folder);self.embed_dim=model.embed_dim;self.device=model.device
        self.image_token_cache=self
    def cache_image_features(self,rgb):return None
    def feature_signature(self):return self.base.feature_signature()
    def set_image_source(self,path,rgb):self.source=Path(path);self.base.set_image_source(path,rgb)
    def extract_features_batch(self,rgb,masks):
        import hashlib
        from tools.run_mainline_full import identity,checked_checkpoint,save_npz,write
        from src.semantic_rescue import mask_hash
        import json
        hashes=[mask_hash(m) for m in masks]
        models=[m for _,m,_ in self.base.models] if hasattr(self.base,'models') else [self.base]
        def receipts():return [identity(m.image_token_cache.folder/(self.source.stem+'.json')) for m in models]
        # First live scale encoding may create these receipts.
        if any(not (m.image_token_cache.folder/(self.source.stem+'.json')).is_file() for m in models):
            for m in models:m.image_token_cache.tokens(m,rgb)
        binding=dict(image=identity(self.source),mask_hashes=hashes,signature=self.feature_signature(),encoder_receipts=receipts())
        name=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        target=self.folder/(self.source.stem+'_'+name+'.npz')
        if checked_checkpoint(target,binding):
            with np.load(target,allow_pickle=False) as f:result=f['features']
        else:
            result=np.asarray(self.base.extract_features_batch(rgb,masks),np.float32).reshape(-1,self.embed_dim)
            save_npz(target,features=result);write(target.with_suffix('.json'),dict(binding=binding,artifact=identity(target)))
        if result.shape!=(len(masks),self.embed_dim) or not np.isfinite(result).all():raise ValueError('Bad query cache')
        return list(result)
    def extract_features(self,rgb,mask):return torch.as_tensor(self.extract_features_batch(rgb,[mask])[0],device=self.device)
    def _warn_feature_failure(self,error):raise RuntimeError('Strict cached feature failure') from error
