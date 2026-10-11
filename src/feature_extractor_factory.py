"""Build frozen SAM3 features, optionally followed by a trained MLP adapter."""

from __future__ import annotations

import torch

from configs.config import (
    FEATURE_ADAPTER_CKPT,
    SAM3_CKPT,
    SAM3_CONTEXT_SUBTRACT,
    SAM3_FEATURE_LEVEL,
    SAM3_FEATURE_SOURCE,
    SAM3_RESOLUTION,
    SAM3_TOKEN_NORM,
    SAM3_USE_BF16,
    SAM3_POOLING,
    SAM3_GMP_RHO,
)


def build_feature_extractor(
    device: torch.device,
    *,
    use_interactivity: bool = False,
):
    """Build SAM3 without loading the independent mask proposer."""
    from src.feature_extractor_sam3 import SAM3FeatureExtractor

    extractor = SAM3FeatureExtractor(
        model_path=SAM3_CKPT,
        device=device,
        resolution=SAM3_RESOLUTION,
        feature_level=SAM3_FEATURE_LEVEL,
        feature_source=SAM3_FEATURE_SOURCE,
        token_norm=SAM3_TOKEN_NORM,
        context_subtract=SAM3_CONTEXT_SUBTRACT,
        use_bf16=SAM3_USE_BF16,
        use_interactivity=use_interactivity,
        pooling=SAM3_POOLING,
        gmp_rho=SAM3_GMP_RHO,
    )
    import os
    cache_dir = os.environ.get('OVERRIDE_IMAGE_TOKEN_CACHE_DIR', '')
    if cache_dir:
        from src.image_token_cache import ImageTokenCache
        extractor.image_token_cache = ImageTokenCache(cache_dir, extractor.feature_signature())
    if FEATURE_ADAPTER_CKPT:
        from src.feature_adapter import AdaptedFeatureExtractor
        return AdaptedFeatureExtractor(extractor, FEATURE_ADAPTER_CKPT)
    return extractor


__all__ = ["build_feature_extractor"]
