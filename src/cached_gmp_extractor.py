"""Read-only cached SAM3 tokens with the exact mainline GMP operation order."""
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from src.hard_center_gmp import aggregate, describe
from src.image_token_cache import ImageTokenCache


class CachedGMPExtractor:
    def __init__(self, folder, signature, device, expected_receipts=None):
        required = dict(backbone='sam3', feature_source='trunk_fpn', feature_level=-1,
                        resolution=1008, token_norm='layernorm', context_subtract=0.0,
                        embed_dim=1280, feature_grid=[72, 72], use_bf16=False,
                        precision_policy='standard_vit_mlp_fp32_v1')
        if (any(signature.get(k) != v for k, v in required.items())
                or signature.get('aggregation') != describe(.1) or signature.get('adapter')):
            raise ValueError('Requires the frozen FP32 mainline SAM3/GMP source signature')
        self._signature = copy.deepcopy(signature)
        self.device = torch.device(device)
        self.embed_dim = 1280
        self.image_token_cache = ImageTokenCache(folder, signature, read_only=True)
        self.expected_receipts = expected_receipts

    def feature_signature(self):
        return copy.deepcopy(self._signature)

    def set_image_source(self, path, rgb):
        if self.expected_receipts is not None:
            from tools.run_mainline_full import identity
            receipt = self.image_token_cache.folder / (Path(path).stem + '.json')
            key = str(receipt.resolve())
            if key not in self.expected_receipts or identity(receipt) != self.expected_receipts[key]:
                raise ValueError(f'Inherited image-token receipt changed: {receipt}')
        self.image_token_cache.bind(path, rgb)

    def cache_image_features(self, rgb):
        return self.image_token_cache.tokens(self, rgb)

    def _preprocess_image(self, _):
        raise RuntimeError('Cached-only extraction must never execute the SAM3 encoder')

    def _forward_tokens(self, _):
        raise RuntimeError('Cached-only extraction must never execute the SAM3 encoder')

    @torch.no_grad()
    def _aggregate_many(self, rgb, masks):
        tokens, grid = self.cache_image_features(rgb)
        values = []
        for mask in masks:
            if np.asarray(mask).shape != rgb.shape[:2]:
                raise ValueError('Mask and original image dimensions differ')
            values.append(aggregate(tokens[0], mask, 1008, grid, .1))
        return list(F.normalize(torch.stack(values).unsqueeze(0).mean(0), dim=1)) if values else []

    def extract_features_batch(self, rgb, masks):
        if not masks:
            return []
        return [x.detach().cpu().numpy() for x in self._aggregate_many(rgb, masks)]

    def extract_features(self, rgb, mask):
        return self._aggregate_many(rgb, [mask])[0]

    def extract_features_for_regions(self, path, regions, min_mask_area=0):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.set_image_source(path, rgb)
        chosen = [r for r in regions if r.get('mask') is not None and r['mask'].sum() >= min_mask_area]
        vectors = self.extract_features_batch(rgb, [r['mask'] for r in chosen])
        output = {}
        for row, vector in zip(chosen, vectors):
            output.setdefault(int(row['cls_id']), []).append(vector)
        return output

    def _warn_feature_failure(self, exc):
        raise RuntimeError('Cached GMP failed; no fallback') from exc
