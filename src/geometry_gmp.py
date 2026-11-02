"""Experimental parameterized hard-center GMP with square/letterbox geometry.

Native mainline implementations and historical encoder signatures stay intact.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from src.hard_center_gmp import uniform_gmp


def layout(shape, resolution, letterbox=True):
    h,w = map(int, shape)
    if min(h,w,resolution) <= 0: raise ValueError('Invalid image geometry')
    if letterbox:
        scale = resolution/max(h,w)
        nh,nw = max(1,round(h*scale)),max(1,round(w*scale))
    else: nh=nw=resolution
    return nh,nw,(resolution-nh)//2,(resolution-nw)//2


def select_centers(mask, resolution=1008, grid=(72,72), letterbox=True):
    m = np.asarray(mask)
    if m.ndim != 2 or min(m.shape)<1 or not np.isin(m,[0,1]).all():
        raise ValueError('Expected binary mask')
    gh,gw = map(int, grid)
    if min(gh,gw,resolution)<=0: raise ValueError('Invalid grid')
    nh,nw,top,left = layout(m.shape,resolution,letterbox)
    yy,xx = np.meshgrid((np.arange(gh)+.5)*resolution/gh,
                        (np.arange(gw)+.5)*resolution/gw,indexing='ij')
    inside = (yy>=top)&(yy<top+nh)&(xx>=left)&(xx<left+nw)
    iy = np.floor((yy-top)*m.shape[0]/nh).astype(int).clip(0,m.shape[0]-1)
    ix = np.floor((xx-left)*m.shape[1]/nw).astype(int).clip(0,m.shape[1]-1)
    indices = np.flatnonzero((inside & m[iy,ix]).ravel())
    if not len(indices) and m.any():
        if not inside.any(): raise ValueError('Letterbox content is narrower than the token grid')
        y,x = np.nonzero(m)
        cy = top+(y.mean()+.5)*nh/m.shape[0]-.5
        cx = left+(x.mean()+.5)*nw/m.shape[1]-.5
        distance = (xx-cx)**2+(yy-cy)**2
        distance[~inside] = np.inf
        indices = np.array([int(distance.argmin())])
    return indices


def describe(resolution=1008, grid=(72,72), rho=.1, letterbox=True):
    from src.hard_center_gmp import describe as native
    return dict(native(rho), version=2, resolution=int(resolution), grid=list(grid),
                geometry='pixel_center_inverse_letterbox_exclude_padding_nearest_content' if letterbox
                         else 'parameterized_square_center_floor_original_nearest_centroid',
                preprocessing='letterbox_center_zero_normalized_padding' if letterbox else 'square')


def make_extractor(checkpoint, device, folder, resolution=1008, read_only=False, signature=None):
    from src.feature_extractor_sam3 import SAM3FeatureExtractor
    from src.image_token_cache import ImageTokenCache
    from torchvision.transforms import v2

    class GeometryExtractor(SAM3FeatureExtractor):
        def feature_signature(self):
            if hasattr(self,'_cached_signature'): return copy.deepcopy(self._cached_signature)
            value = super().feature_signature()
            value['aggregation'] = describe(self.resolution,self.feature_grid,self.gmp_rho)
            value['preprocessing_geometry'] = dict(mode='letterbox',padding='normalized_zero',
                                                   placement='center',resize='rounded_aspect_ratio')
            return value

        def _preprocess_image(self, rgb):
            if read_only: raise RuntimeError('Missing read-only letterbox cache; no encoding fallback')
            nh,nw,top,left = layout(rgb.shape[:2],self.resolution)
            image = v2.functional.to_image(rgb).to(self.device)
            image = v2.ToDtype(torch.uint8,scale=True)(image)
            image = v2.Resize((nh,nw))(image)
            image = v2.Normalize([.5]*3,[.5]*3)(v2.ToDtype(torch.float32,scale=True)(image))
            canvas = image.new_zeros((3,self.resolution,self.resolution))
            canvas[:,top:top+nh,left:left+nw] = image
            return canvas[None],(top,left,nh,nw)

        def _aggregate_many(self,rgb,masks):
            tokens,grid = self.image_token_cache.tokens(self,rgb)
            values = []
            for mask in masks:
                if np.asarray(mask).shape != rgb.shape[:2]: raise ValueError('Mask/image mismatch')
                indices = select_centers(mask,self.resolution,grid)
                values.append(uniform_gmp(tokens[0],indices,self.gmp_rho))
            return list(F.normalize(torch.stack(values),dim=1)) if values else []

    if read_only:
        if not signature or signature.get('aggregation') != describe(resolution,signature['feature_grid'],.1):
            raise ValueError('Incompatible letterbox feature signature')
        model = GeometryExtractor.__new__(GeometryExtractor)
        model._cached_signature = copy.deepcopy(signature)
        model.device=torch.device(device);model.resolution=resolution;model.feature_grid=signature['feature_grid']
        model.gmp_rho=.1;model.embed_dim=signature['embed_dim'];model.pooling='hard_center_gmp'
    else:
        model = GeometryExtractor(model_path=checkpoint,device=torch.device(device),resolution=resolution,
            feature_source='trunk_fpn',feature_level=-1,token_norm='layernorm',context_subtract=0.,
            use_bf16=False,enable_segmentation=False,use_interactivity=False,pooling='smfa',gmp_rho=.1)
        model.pooling='hard_center_gmp'
    model.image_token_cache=ImageTokenCache(folder,model.feature_signature(),read_only=read_only)
    return model
