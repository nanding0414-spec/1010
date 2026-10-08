"""Persist and validate the exact feature space used by a feature bank."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


FEATURE_SIGNATURE_FILENAME = "feature_signature.json"
REQUIRED_SIGNATURE_KEYS = {
    "format_version",
    "backbone",
    "feature_source",
    "feature_level",
    "resolution",
    "token_norm",
    "context_subtract",
    "embed_dim",
    "feature_grid",
    "use_bf16",
    "model_name",
}


def save_feature_signature(meta_dir: str | Path, signature: Dict[str, Any]) -> Path:
    missing = REQUIRED_SIGNATURE_KEYS - set(signature)
    if missing:
        raise ValueError(f"Feature signature is missing keys: {sorted(missing)}")
    path = Path(meta_dir) / FEATURE_SIGNATURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(signature, f, indent=2, sort_keys=True)
    return path


def load_feature_signature(meta_dir: str | Path) -> Dict[str, Any] | None:
    path = Path(meta_dir) / FEATURE_SIGNATURE_FILENAME
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        signature = json.load(f)
    if not isinstance(signature, dict):
        raise ValueError(f"Invalid feature signature in {path}")
    return signature


def validate_feature_signature(
    meta_dir: str | Path,
    expected: Dict[str, Any],
    allow_missing: bool = False,
) -> None:
    actual = load_feature_signature(meta_dir)
    if actual is None:
        if allow_missing:
            print("[FeatureBank] Warning: legacy bank has no feature signature")
            return
        raise RuntimeError(
            f"Feature bank under {meta_dir} has no {FEATURE_SIGNATURE_FILENAME}. "
            "Rebuild it with run_build_feature_bank.py, or explicitly set "
            "OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK=true for a legacy bank."
        )

    mismatches = []
    for key in sorted(REQUIRED_SIGNATURE_KEYS | {"adapter", "pooling", "aggregation", "resolution_adaptation", "precision_policy", "checkpoint_sha256", "experimental_pooling"}):
        if actual.get(key) != expected.get(key):
            mismatches.append(
                f"{key}: bank={actual.get(key)!r}, inference={expected.get(key)!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Feature bank and inference extractor use different feature spaces:\n"
            f"  - {details}\nRebuild the bank or restore matching feature-model settings."
        )


__all__ = [
    "FEATURE_SIGNATURE_FILENAME",
    "load_feature_signature",
    "save_feature_signature",
    "validate_feature_signature",
]
