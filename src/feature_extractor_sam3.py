"""Frozen SAM3 feature extractor for feature bank building and inference.

Uses the official `sam3` package (facebookresearch/sam3):
    pip install git+https://github.com/facebookresearch/sam3.git
    hf auth login   # facebook/sam3 checkpoint is gated, needs access approval

How it plugs into ModuSeg:
  * `trunk`: raw 1024-D final ViT tokens (stride 14).
  * `fpn`: one 256-D segmentation-neck level.
  * `trunk_fpn`: normalized raw ViT and FPN maps concatenated channel-wise.

The official image builder creates FPN strides [3.5, 7, 14, 28] and then
removes stride 28 with `scalp=1`. Therefore the visible levels are
[-3=stride3.5, -2=stride7, -1=stride14]. Masks stay at original resolution and
the base SMFA maps them directly to the selected feature grid with INTER_AREA.
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from configs.config import (
    SAM3_CONTEXT_SUBTRACT,
    SAM3_FEATURE_SOURCE,
    SAM3_TOKEN_NORM,
)
from src.feature_extractor import BaseFeatureExtractor
from src.sam3_runtime import build_sam3_model


class SAM3FeatureExtractor(BaseFeatureExtractor):
    """SAM3 vision backbone as a drop-in BaseFeatureExtractor subclass.

    Args:
        model_path: local sam3.pt checkpoint. None -> auto-download from
            HuggingFace (facebook/sam3, requires `hf auth login` + access).
        resolution: Default 1008. Other multiples of 336 use experimental
            feature-only global RoPE adaptation; not a segmentation-head mode.
        feature_level: index into the visible FPN. -1=stride14, -2=stride7,
            -3=stride3.5. Must match between bank build and inference.
        feature_source: `trunk`, `fpn`, or `trunk_fpn`.
        token_norm: `layernorm`, `l2`, or `none`, applied before SMFA.
        use_bf16: run backbone forward under bfloat16 autocast.
        enable_segmentation: keep SAM3's segmentation head (True is required by
            the pseudo-mask generator; the feature extractor itself only needs
            the backbone, so False saves VRAM).
    """

    def __init__(
        self,
        model_path=None,
        device=None,
        resolution=1008,
        feature_level=-1,
        feature_source=None,
        token_norm=None,
        context_subtract=None,
        use_bf16=False,
        enable_segmentation=True,
        use_interactivity=True,
        query_chunk_size=256,
        pooling='hard_center_gmp',
        gmp_rho=.1,
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device  # 必须在 super().__init__() 之前就设好（model.to / _get_embed_dim 都会用到）
        self.resolution = resolution
        self.feature_level = feature_level
        self.feature_source = (feature_source or SAM3_FEATURE_SOURCE).lower()
        self.token_norm = (token_norm or SAM3_TOKEN_NORM).lower()
        self.context_subtract = float(
            SAM3_CONTEXT_SUBTRACT if context_subtract is None else context_subtract
        )
        self.use_bf16 = use_bf16
        self.feature_grid = None
        self.pooling = pooling
        self.gmp_rho = float(gmp_rho)
        if pooling not in {'hard_center_gmp', 'smfa'}:
            raise ValueError('Unsupported pooling')
        if pooling == 'hard_center_gmp' and (resolution != 1008 or use_bf16 or self.context_subtract != 0):
            raise ValueError('Mainline GMP requires 1008, FP32, and context_subtract=0')
        if model_path and Path(model_path).is_file():
            digest = hashlib.sha256()
            with Path(model_path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            self.checkpoint_sha256 = digest.hexdigest()
        else:
            raise FileNotFoundError('A local SAM3 checkpoint is required to bind feature identity')

        if self.feature_source not in {"trunk", "fpn", "trunk_fpn"}:
            raise ValueError(
                f"Unknown SAM3 feature source: {self.feature_source}. "
                "Expected trunk, fpn, or trunk_fpn."
            )
        if self.token_norm not in {"none", "layernorm", "l2"}:
            raise ValueError(
                f"Unknown SAM3 token norm: {self.token_norm}. "
                "Expected none, layernorm, or l2."
            )
        if not 0.0 <= self.context_subtract <= 1.0:
            raise ValueError("SAM3 context subtraction must be in [0, 1]")

        if resolution != 1008 and (resolution < 336 or resolution % 336):
            raise ValueError('Experimental feature resolution must be a multiple of 336')
        if resolution != 1008 and use_interactivity:
            raise ValueError('Non-native resolution is supported only for feature extraction; disable interactivity')
        # SAM3 is only selected when this class is explicitly requested.
        # Its loader is shared with the independent online mask proposer.
        model = build_sam3_model(
            device=self.device,
            checkpoint_path=model_path,
            enable_segmentation=enable_segmentation if resolution == 1008 else False,
            enable_inst_interactivity=use_interactivity,
        )
        self.model = model  # 先挂到 self 上，供 _get_embed_dim 等使用
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        from src.sam3_resolution import adapt_feature_resolution
        self.resolution_adaptation = adapt_feature_resolution(
            self.model.backbone.vision_backbone.trunk, resolution)
        if resolution != 1008:
            from src.sam3_resolution import enable_query_chunking
            enable_query_chunking(self.model.backbone.vision_backbone.trunk, query_chunk_size)
            self.resolution_adaptation['query_chunk_size'] = query_chunk_size
            print(f'[SAM3] global attention query chunk={query_chunk_size} (0=original SDPA)', flush=True)

        # 与 Sam3Processor 完全相同的预处理（方形缩放 + [0,1] + mean/std 0.5）
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        embed_dim = self._get_embed_dim()
        super().__init__(
            device=device,
            patch_size=(16, 16),  # 仅占位；SAM3 路径用实际特征图尺寸
            embed_dim=embed_dim,
            model_prefix="SAM3",
            norm_mean=(0.0, 0.0, 0.0),
            norm_std=(1.0, 1.0, 1.0),
            context_subtract=self.context_subtract,
        )
        self.model = model  # 基类 __init__ 会把 self.model 重置为 None，这里恢复
        grid_text = "unknown" if self.feature_grid is None else f"{self.feature_grid[0]}x{self.feature_grid[1]}"
        stride_text = (
            "unknown"
            if self.feature_grid is None
            else f"{self.resolution / self.feature_grid[0]:g}"
        )
        print(
            f"[SAM3] resolution={resolution}, source={self.feature_source}, "
            f"feature_level={feature_level}, grid={grid_text}, stride={stride_text}, "
            f"embed_dim={embed_dim}, token_norm={self.token_norm}, "
            f"context_subtract={self.context_subtract:g}, use_bf16={use_bf16}"
        )
        print(f"[SAM3] pooling={self.pooling}, rho={self.gmp_rho:g}")

    # ------------------------------------------------------------------ #
    # Base class overrides
    # ------------------------------------------------------------------ #

    def _preprocess_image(self, image_rgb):
        """SAM3 方形缩放预处理；mask 保持原分辨率（无 padding）。"""
        h, w = image_rgb.shape[:2]
        img = v2.functional.to_image(image_rgb).to(self.device)  # uint8 -> float[0,1]
        x = self.transform(img).unsqueeze(0)                     # [1,3,R,R] in [-1,1]
        return x, (0, 0, h, w)

    @staticmethod
    def _unwrap_feature(feats):
        return feats.tensors if hasattr(feats, "tensors") else feats

    def _normalize_feature_map(self, feats):
        feats = feats.float()
        if self.token_norm == "layernorm":
            feats = feats.permute(0, 2, 3, 1)
            feats = F.layer_norm(feats, (feats.shape[-1],))
            return feats.permute(0, 3, 1, 2).contiguous()
        if self.token_norm == "l2":
            return F.normalize(feats, p=2, dim=1)
        return feats

    def _select_fpn_conv(self, vision_neck):
        scalp = int(getattr(self.model.backbone, "scalp", 0))
        visible_count = len(vision_neck.convs) - scalp
        level = self.feature_level
        if level < 0:
            level += visible_count
        if level < 0 or level >= visible_count:
            raise IndexError(
                f"SAM3 feature_level={self.feature_level} is invalid for "
                f"{visible_count} visible FPN levels"
            )
        return vision_neck.convs[level]

    def _forward_tokens(self, inp):
        """Forward the ViT once, then select/fuse the configured feature maps."""
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if (self.use_bf16 and inp.is_cuda)
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            vision_neck = self.model.backbone.vision_backbone
            trunk_outputs = vision_neck.trunk(inp)
            trunk_feats = self._unwrap_feature(trunk_outputs[-1])

            if self.feature_source == "trunk":
                feats = self._normalize_feature_map(trunk_feats)
            else:
                fpn_feats = self._select_fpn_conv(vision_neck)(trunk_feats)
                fpn_feats = self._normalize_feature_map(fpn_feats)
                if self.feature_source == "fpn":
                    feats = fpn_feats
                else:
                    trunk_norm = self._normalize_feature_map(trunk_feats)
                    if trunk_norm.shape[-2:] != fpn_feats.shape[-2:]:
                        trunk_norm = F.interpolate(
                            trunk_norm,
                            size=fpn_feats.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    # Equalize source energy so the 1024-D trunk does not drown
                    # out the 256-D FPN solely because it has more channels.
                    trunk_norm = F.normalize(trunk_norm, p=2, dim=1)
                    fpn_feats = F.normalize(fpn_feats, p=2, dim=1)
                    feats = torch.cat([trunk_norm, fpn_feats], dim=1)

        feats = feats.float()
        B, C, H, W = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)

    def _fallback_global_feature(self, inp):
        tokens, _ = self._forward_tokens(inp)
        return torch.nn.functional.normalize(tokens.squeeze(0).mean(0), p=2, dim=0)

    def _get_embed_dim(self):
        try:
            dummy = torch.zeros(
                1, 3, self.resolution, self.resolution, device=self.device
            )
            feats, grid = self._forward_tokens(dummy)
            self.feature_grid = tuple(grid)
            return int(feats.shape[-1])
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:  # pragma: no cover - requires a SAM3 runtime
            raise RuntimeError(
                "SAM3 feature probe failed. Check that the installed SAM3 version "
                "matches this project and that the selected feature level is valid."
            ) from exc

    def feature_signature(self):
        """Stable description of the feature space stored with the bank."""
        signature = {
            "format_version": 3,
            "backbone": "sam3",
            "feature_source": self.feature_source,
            "feature_level": int(self.feature_level),
            "resolution": int(self.resolution),
            "token_norm": self.token_norm,
            "context_subtract": float(self.context_subtract),
            "embed_dim": int(self.embed_dim),
            "feature_grid": list(self.feature_grid) if self.feature_grid else None,
            "use_bf16": bool(self.use_bf16),
            "model_name": "sam3",
            "precision_policy": "standard_vit_mlp_bf16_autocast_v1" if self.use_bf16 else "standard_vit_mlp_fp32_v1",
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        if getattr(self, 'resolution_adaptation', None) is not None:
            signature['resolution_adaptation'] = self.resolution_adaptation
        if self.pooling == 'hard_center_gmp':
            from src.hard_center_gmp import describe
            signature['aggregation'] = describe(self.gmp_rho)
        return signature

    def _aggregate_many(self, image_rgb, masks):
        from src.hard_center_gmp import aggregate
        inp, _ = self._preprocess_image(image_rgb)
        tokens, grid = self._forward_tokens(inp)
        if tokens is None or tokens.shape[0] != 1:
            raise RuntimeError('Expected one-image spatial tokens')
        results = []
        for mask in masks:
            if np.asarray(mask).shape != image_rgb.shape[:2]:
                raise ValueError('Mask and original image dimensions differ')
            results.append(aggregate(tokens[0], mask, self.resolution, grid, self.gmp_rho))
        # The historical single-phase route normalizes again after phase fusion.
        return list(F.normalize(torch.stack(results).unsqueeze(0).mean(0), dim=1)) if results else []

    def extract_features(self, image_rgb, mask_bool):
        if self.pooling == 'smfa':
            return super().extract_features(image_rgb, mask_bool)
        return self._aggregate_many(image_rgb, [mask_bool])[0]

    def extract_features_batch(self, image_rgb, masks_bool_list):
        if self.pooling == 'smfa':
            return super().extract_features_batch(image_rgb, masks_bool_list)
        if not masks_bool_list:
            return []
        return [x.detach().cpu().numpy() for x in self._aggregate_many(image_rgb, masks_bool_list)]

    def extract_features_for_regions(self, img_path, regions, min_mask_area=0):
        if self.pooling == 'smfa':
            return super().extract_features_for_regions(img_path, regions, min_mask_area)
        import cv2
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        chosen = [r for r in regions if r.get('mask') is not None and r['mask'].sum() >= min_mask_area]
        vectors = self.extract_features_batch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [r['mask'] for r in chosen])
        output = {}
        for region, vector in zip(chosen, vectors):
            if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6:
                raise ValueError('Nonempty training region produced an invalid/zero GMP vector')
            output.setdefault(int(region['cls_id']), []).append(vector)
        return output


__all__ = ["SAM3FeatureExtractor"]
