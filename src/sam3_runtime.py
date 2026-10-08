"""Shared SAM3 model loader for pseudo masks and online mask proposals."""

from __future__ import annotations

import os

import torch

from configs.config import SAM3_CKPT


def build_sam3_model(
    device: torch.device | None = None,
    *,
    checkpoint_path: str | None = None,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
):
    """Build the SAM3 model used by the mask-producing parts of the pipeline."""
    from sam3.model_builder import build_sam3_image_model

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = checkpoint_path or SAM3_CKPT
    local_ckpt = model_path if model_path and os.path.exists(model_path) else None
    if local_ckpt:
        print(f"[SAM3] Loading local checkpoint: {local_ckpt}")
    else:
        print("[SAM3] Local checkpoint not found, downloading from HuggingFace...")

    model = build_sam3_image_model(
        checkpoint_path=local_ckpt,
        load_from_HF=(local_ckpt is None),
        device="cuda" if device.type == "cuda" else "cpu",
        eval_mode=True,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    model = model.to(device).eval()
    # Fix vendor MLP's internal BF16 casts before any feature/proposal probe.
    from src.sam3_precision import use_standard_vit_mlp
    count = use_standard_vit_mlp(model)
    if not count:
        raise RuntimeError('No supported SAM3 ViT MLP found for the FP32 policy')
    model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


__all__ = ["build_sam3_model"]
