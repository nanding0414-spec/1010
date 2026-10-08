"""Learn a retrieval embedding after frozen backbone + SMFA pooling."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, num_layers=2):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError("num_layers must be 2 or 3")
        self.architecture = dict(input_dim=int(input_dim), hidden_dim=int(hidden_dim),
                                 output_dim=int(output_dim))
        # Preserve the architecture signature of existing two-layer banks.
        if num_layers != 2:
            self.architecture["num_layers"] = num_layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        if num_layers == 3:
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(F.normalize(x.float(), dim=-1)), dim=-1)


class CosineClassifier(nn.Module):
    """Train embeddings using angular class scores, matching cosine retrieval."""
    def __init__(self, dimension, num_classes, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dimension))
        nn.init.normal_(self.weight, std=0.01)
        self.scale = float(scale)

    def forward(self, z):
        return self.scale * F.linear(F.normalize(z, dim=-1),
                                     F.normalize(self.weight, dim=-1))


def feature_consistency_loss(model, x, clean_z, drop_probability):
    """Feature dropout is an embedding perturbation, not a simulated mask edit."""
    if not 0 <= drop_probability < 1:
        raise ValueError("feature dropout must be in [0, 1)")
    perturbed = F.dropout(x, p=drop_probability, training=True)
    perturbed_z = model(perturbed)
    valid = perturbed.norm(dim=-1) > 1e-6
    if not valid.any():
        return perturbed_z.sum() * 0
    target = F.normalize(clean_z.detach(), dim=-1)
    return (1 - (perturbed_z[valid] * target[valid]).sum(-1)).mean()


def supervised_contrastive_loss(z, labels, temperature=0.1, background_index=None):
    """Exclude self matches; background is negative-only, not one compact cluster."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(z) < 2:
        return z.sum() * 0
    self_mask = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    if background_index is not None:
        positive &= labels[:, None].ne(background_index)
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return z.sum() * 0
    logits = F.normalize(z, dim=1) @ F.normalize(z, dim=1).T / temperature
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = log_probs.masked_fill(~positive, 0).sum(1)
    return -(positive_log_probs[valid] / counts[valid]).mean()


def load_adapter(path, device="cpu", source_signature=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported adapter checkpoint format")
    expected = payload["source_signature"]
    if expected.get('experimental_pooling'):
        raise ValueError('Experimental adapters require verified mainline import')
    aggregation = expected.get('aggregation')
    if aggregation:
        from src.hard_center_gmp import describe
        if aggregation != describe(aggregation.get('rho', .1)):
            raise ValueError('Unsupported adapter aggregation signature')
    if expected.get('pooling'):
        raise ValueError('Unsupported legacy pooling signature')
    if expected.get("backbone") != "sam3" or expected.get("adapter"):
        raise ValueError("Adapter must be trained on unadapted SAM3 features")
    if source_signature is not None and expected != source_signature:
        raise ValueError("Adapter source signature differs from current SAM3 feature settings")
    architecture = dict(payload["architecture"])
    # Checkpoints predating configurable depth always contain two Linear layers.
    architecture.setdefault("num_layers", 2)
    adapter = FeatureAdapter(**architecture)
    if adapter.architecture["input_dim"] != expected["embed_dim"]:
        raise ValueError("Adapter input dimension differs from its source signature")
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.to(device).eval().requires_grad_(False)
    return adapter, payload


def adapted_signature(source, checkpoint_path, architecture):
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    signature = dict(source)
    signature["embed_dim"] = architecture["output_dim"]
    signature["adapter"] = {
        "type": "gmp_mlp_v1" if source.get('aggregation') else "smfa_mlp_v1", "sha256": digest.hexdigest(),
        "architecture": architecture, "input_norm": "l2", "output_norm": "l2",
    }
    return signature


@torch.inference_mode()
def project_features(adapter, features, device, batch_size=1024):
    """Preserve invalid/empty SMFA features as zeros instead of mapping MLP bias."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != adapter.architecture["input_dim"]:
        raise ValueError(f"Unexpected input feature shape: {x.shape}")
    output = np.zeros((len(x), adapter.architecture["output_dim"]), dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 1e-6))
    for start in range(0, len(valid), batch_size):
        rows = valid[start:start + batch_size]
        output[rows] = adapter(torch.from_numpy(x[rows]).to(device)).cpu().numpy()
    return output


class AdaptedFeatureExtractor:
    """Wrap all public SMFA extraction paths with the exact same projection."""
    def __init__(self, base, checkpoint_path):
        self.base = base
        self.device = base.device
        self.adapter, _ = load_adapter(checkpoint_path, self.device, base.feature_signature())
        self.embed_dim = self.adapter.architecture["output_dim"]
        self._signature = adapted_signature(base.feature_signature(), checkpoint_path,
                                            self.adapter.architecture)

    def feature_signature(self):
        return self._signature

    def _project(self, features):
        return project_features(self.adapter, features, self.device)

    def extract_features(self, image_rgb, mask_bool):
        vector = self.base.extract_features(image_rgb, mask_bool).detach().cpu().numpy()
        return torch.from_numpy(self._project(vector[None])[0]).to(self.device)

    def extract_features_batch(self, image_rgb, masks_bool_list):
        vectors = self.base.extract_features_batch(image_rgb, masks_bool_list)
        return list(self._project(np.stack(vectors))) if vectors else []

    def extract_features_for_regions(self, img_path, regions, min_mask_area=0):
        vectors = self.base.extract_features_for_regions(img_path, regions, min_mask_area)
        return {key: list(self._project(np.stack(value))) for key, value in vectors.items()
                if value}

    def _warn_feature_failure(self, exc):
        self.base._warn_feature_failure(exc)
