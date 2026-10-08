"""Base class for visual feature extractors using mask-weighted average pooling."""
import os
from typing import List, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from configs.config import MIN_MASK_AREA


class BaseFeatureExtractor(ABC):
    """Base visual feature extractor using mask pooling strategy."""

    def __init__(
        self,
        device: torch.device,
        patch_size: Tuple[int, int],
        embed_dim: int,
        model_prefix: str,
        norm_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        context_subtract: float = 0.0,
    ) -> None:
        self.device = device
        self.model_prefix = model_prefix
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_mean = torch.tensor(norm_mean).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(norm_std).view(1, 3, 1, 1)
        self.context_subtract = float(context_subtract)
        self._feature_failure_count = 0
        self.model = None  # initialized by subclass

    def _preprocess_image(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Pad image to patch_size multiples without rescaling.
        Returns (tensor [1,3,H,W], (pad_h, pad_w, h, w))."""
        h, w = image_rgb.shape[:2]
        ph, pw = self.patch_size
        
        new_h, new_w = h, w
        img_resized = image_rgb.copy()
        pad_h = (ph - new_h % ph) % ph
        pad_w = (pw - new_w % pw) % pw

        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        img = img_padded.astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        x = (x - self.norm_mean) / self.norm_std
        
        return x.to(self.device), (pad_h, pad_w, new_h, new_w)

    def _get_patch_grid(self, inp: torch.Tensor) -> Tuple[int, int]:
        """Compute patch grid size from input tensor."""
        _, _, h, w = inp.shape
        ph, pw = self.patch_size
        return h // ph, w // pw

    def _tokens_from_forward(self, feats: Union[torch.Tensor, dict]) -> Optional[torch.Tensor]:
        """Parse [B, N, C] token sequence from model output."""
        if isinstance(feats, dict):
            for k in ['x_norm', 'xnorm', 'x', 'tokens', 'last_hidden_state']:
                if k in feats and isinstance(feats[k], torch.Tensor) and feats[k].dim() == 3:
                    return feats[k]
            for v in feats.values():
                if isinstance(v, torch.Tensor) and v.dim() == 3:
                    return v
            return None
        if isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                return feats
            if feats.dim() == 4:
                # [B, C, H, W] -> [B, H*W, C]
                return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], -1, feats.shape[1])
            return None
        return None

    @abstractmethod
    def _forward_tokens(self, inp: torch.Tensor) -> Tuple[Optional[torch.Tensor], Tuple[int, int]]:
        """Forward pass returning token sequence and patch grid size. Implemented by subclasses."""
        pass

    def _downsample_mask_to_patches(
        self, 
        mask_bool: np.ndarray, 
        Hp: int, 
        Wp: int,
        transform_info: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Downsample mask to patch grid size for feature aggregation."""
        pad_h, pad_w, resized_h, resized_w = transform_info
        
        mask_float = mask_bool.astype(np.float32)

        mask_resized = cv2.resize(
            mask_float, 
            (resized_w, resized_h), 
            interpolation=cv2.INTER_NEAREST
        )

        mask_padded = cv2.copyMakeBorder(
            mask_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )

        mask_patch = cv2.resize(mask_padded, (Wp, Hp), interpolation=cv2.INTER_AREA)

        return mask_patch.reshape(-1)

    def extract_features(self, image_rgb: np.ndarray, mask_bool: np.ndarray) -> torch.Tensor:
        """Extract single-region feature via mask-weighted average pooling. Returns [C] vector."""
        return self._extract_mask_pooling_feature(image_rgb, mask_bool)

    def _warn_feature_failure(self, exc: Exception) -> None:
        """Report a few extraction failures without flooding long dataset runs."""
        self._feature_failure_count += 1
        if self._feature_failure_count <= 5:
            print(f"[{self.model_prefix}] feature extraction failed: {exc}")
        elif self._feature_failure_count == 6:
            print(f"[{self.model_prefix}] further feature extraction failures are suppressed")

    def _pool_tokens(self, tok: torch.Tensor, mask_vec: np.ndarray) -> torch.Tensor:
        """Pool [N, C] tokens with soft mask weights and optional context removal."""
        if tok.ndim != 2:
            raise ValueError(f"Expected [N, C] tokens, got {tuple(tok.shape)}")
        if mask_vec.shape[0] != tok.shape[0]:
            raise ValueError(
                f"Mask/token length mismatch: {mask_vec.shape[0]} != {tok.shape[0]}"
            )

        w = torch.from_numpy(mask_vec.astype(np.float32, copy=False)).to(
            device=tok.device, dtype=tok.dtype
        ).view(-1, 1)
        weight_sum = w.sum()
        if weight_sum < 1e-6:
            return torch.zeros(tok.shape[1], device=tok.device, dtype=tok.dtype)

        feat = (tok * w).sum(dim=0) / (weight_sum + 1e-8)
        if self.context_subtract > 0:
            context_w = (1.0 - w).clamp_min(0.0)
            context_sum = context_w.sum()
            if context_sum >= 1e-6:
                context = (tok * context_w).sum(dim=0) / (context_sum + 1e-8)
                feat = feat - self.context_subtract * context
        return F.normalize(feat, p=2, dim=0)

    def _extract_mask_pooling_feature(self, image_rgb: np.ndarray, mask_bool: np.ndarray) -> torch.Tensor:
        """Mask pooling: weighted average of patch tokens by downsampled mask."""
        inp, transform_info = self._preprocess_image(image_rgb)
        tokens, (Hp, Wp) = self._forward_tokens(inp)
        
        if tokens is None:
            raise RuntimeError('SMFA requires spatial tokens; global-pooling fallback is not supported')

        B, N, C = tokens.shape
        assert B == 1

        mask_vec = self._downsample_mask_to_patches(mask_bool, Hp, Wp, transform_info)
        m = mask_vec.astype(np.float32)

        if m.sum() < 1e-6:
            return torch.zeros(C, device=self.device)
            
        tok = tokens.reshape(N, C)
        return self._pool_tokens(tok, m)

    def extract_features_for_regions(
        self,
        img_path: str,
        regions: List[Dict],
        min_mask_area: int = MIN_MASK_AREA,
    ) -> Dict[int, List[np.ndarray]]:
        """Extract features for all regions in an image. Returns {cls_id: [feat_vec, ...]}."""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {}
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        inp, transform_info = self._preprocess_image(image_rgb)
        tokens, (Hp, Wp) = self._forward_tokens(inp)

        if tokens is None:
            raise RuntimeError('SMFA requires spatial tokens; global-pooling fallback is not supported')

        B, N, C = tokens.shape
        assert B == 1
        tok = tokens.reshape(N, C)  # [N, C]

        out: Dict[int, List[np.ndarray]] = {}
        for r in regions:
            cid = int(r.get('cls_id', -1))
            mask = r.get('mask', None)
            if mask is None or mask.sum() < min_mask_area:
                continue
            try:
                m_vec = self._downsample_mask_to_patches(
                    mask.astype(bool), Hp, Wp, transform_info
                ).astype(np.float32)
                if m_vec.sum() < 1e-6:
                    continue
                feat = self._pool_tokens(tok, m_vec)
                feat_np = feat.detach().cpu().numpy()
                out.setdefault(cid, []).append(feat_np)
            except Exception as exc:
                self._warn_feature_failure(exc)
                continue
        return out

    def extract_features_batch(
        self,
        image_rgb: np.ndarray,
        masks_bool_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Batch feature extraction for multiple masks sharing one forward pass."""
        if len(masks_bool_list) == 0:
            return []
        
        inp, transform_info = self._preprocess_image(image_rgb)
        tokens, (Hp, Wp) = self._forward_tokens(inp)
        
        if tokens is None:
            raise RuntimeError('SMFA requires spatial tokens; global-pooling fallback is not supported')

        B, N, C = tokens.shape
        assert B == 1
        tok = tokens.reshape(N, C)

        mask_weights = []
        valid_indices = []
        
        for idx, m in enumerate(masks_bool_list):
            try:
                m_vec = self._downsample_mask_to_patches(
                    m.astype(bool), Hp, Wp, transform_info
                ).astype(np.float32)
                if m_vec.sum() >= 1e-6:
                    mask_weights.append(m_vec)
                    valid_indices.append(idx)
            except Exception as exc:
                self._warn_feature_failure(exc)
        
        feats_out: List[np.ndarray] = [np.zeros(self.embed_dim, dtype=np.float32) for _ in masks_bool_list]
        
        if len(mask_weights) == 0:
            return feats_out
        
        weights_matrix = np.vstack(mask_weights)
        weights_tensor = torch.from_numpy(weights_matrix).to(
            device=tok.device, dtype=tok.dtype
        )

        weighted_sum = torch.matmul(weights_tensor, tok)
        weight_sums = weights_tensor.sum(dim=1, keepdim=True)
        features = weighted_sum / (weight_sums + 1e-8)
        if self.context_subtract > 0:
            context_weights = (1.0 - weights_tensor).clamp_min(0.0)
            context_sums = context_weights.sum(dim=1, keepdim=True)
            valid_context = context_sums >= 1e-6
            context = torch.matmul(context_weights, tok) / (context_sums + 1e-8)
            features = features - self.context_subtract * context * valid_context
        features = F.normalize(features, p=2, dim=1)
        features_np = features.detach().cpu().numpy()

        for i, idx in enumerate(valid_indices):
            feats_out[idx] = features_np[i]
            
        return feats_out


__all__ = ["BaseFeatureExtractor"]
