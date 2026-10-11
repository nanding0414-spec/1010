"""Lossless original-image SAM3 trunk/FPN token cache, independent of masks/GMP."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


class ImageTokenCache:
    def __init__(self, folder, signature, *, read_only=False):
        self.folder = Path(folder)
        self.signature = {k:v for k,v in signature.items() if k not in ('aggregation','adapter')}
        self.binding = None
        self.live = None
        self.read_only = read_only

    def bind(self, image_path, rgb):
        path = Path(image_path).resolve()
        binding = dict(format_version=1, source=str(path), source_sha256=digest(path),
            original_shape=list(rgb.shape[:2]), rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            encoder_signature=self.signature, representation='post_token_normalization_trunk_fpn_before_GMP', dtype='float32')
        if self.binding != binding:
            self.live = None
        self.binding = binding

    def tokens(self, model, rgb):
        import torch
        if self.binding is None:
            raise RuntimeError('Set the source image before using its token cache')
        if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() != self.binding['rgb_sha256']:
            raise ValueError('Token cache source differs from feature image')
        if self.live is not None:
            return self.live
        if not self.read_only:
            self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder/(Path(self.binding['source']).stem+'.npz')
        receipt = target.with_suffix('.json')
        if target.exists() or receipt.exists():
            if not target.is_file() or not receipt.is_file():
                raise ValueError(f'Incomplete token cache pair retained: {target}')
            metadata = json.loads(receipt.read_text())
            if metadata['binding'] != self.binding or metadata['sha256'] != digest(target):
                raise ValueError(f'Token cache signature/source/content changed: {target}')
            with np.load(target, allow_pickle=False) as stored:
                array = stored['tokens']; grid = tuple(map(int, stored['grid']))
            if (array.dtype != np.float32 or array.shape != (*grid, self.signature['embed_dim'])
                    or list(grid) != self.signature['feature_grid'] or not np.isfinite(array).all()):
                raise ValueError('Invalid cached token geometry/dtype')
            tokens = torch.from_numpy(array.reshape(1,-1,array.shape[-1]).copy()).to(model.device)
        else:
            if self.read_only:
                raise FileNotFoundError(f'Required read-only token cache is missing: {target}')
            reserve = int(os.environ.get('MODUSEG_MIN_FREE_BYTES', str(5*2**30)))
            if shutil.disk_usage(self.folder).free < reserve + 72*72*1280*4:
                raise OSError('Insufficient space for image tokens and reserve')
            inp, _ = model._preprocess_image(rgb)
            tokens, grid = model._forward_tokens(inp)
            if tokens.shape != (1,grid[0]*grid[1],self.signature['embed_dim']) or not torch.isfinite(tokens).all():
                raise ValueError('Invalid SAM3 token output')
            array = tokens[0].detach().float().cpu().numpy().reshape(*grid,-1)
            temporary = target.with_name(target.name+'.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,tokens=array,grid=np.asarray(grid,np.int16))
            os.replace(temporary,target)
            metadata = dict(binding=self.binding,sha256=digest(target),shape=list(array.shape),bytes=target.stat().st_size)
            temporary = receipt.with_name(receipt.name+'.tmp')
            temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
            os.replace(temporary,receipt)
        self.live = tokens, grid
        return self.live
