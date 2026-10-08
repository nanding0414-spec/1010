"""Cached-feature data handling and held-out retrieval diagnostics."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from src.feature_bank_storage import load_feature_matrix_npz
from src.feature_signature import load_feature_signature


def image_id_from_sample(sample_id):
    # FeatureBank appends a per-class six-digit counter to the image stem.
    image_id, separator, counter = str(sample_id).rpartition("_")
    if not separator or not counter.isdigit() or len(counter) < 6:
        raise ValueError(f"Expected image_id_counter sample ID, got {sample_id!r}")
    return image_id


def load_training_bank(root, train_list):
    root = Path(root)
    signature = load_feature_signature(root / "meta")
    if not signature or signature.get("backbone") != "sam3" or signature.get("adapter"):
        raise ValueError("Source bank must have an unadapted SAM3 feature signature")
    if signature.get('pooling') or signature.get('experimental_pooling'):
        raise ValueError('Experimental/legacy aggregation banks require explicit verified import')
    if signature.get('aggregation'):
        from src.hard_center_gmp import describe
        if signature['aggregation'] != describe(signature['aggregation'].get('rho', .1)):
            raise ValueError('Unsupported source aggregation signature')
    with (root / "meta" / "class_mapping.json").open(encoding="utf-8") as stream:
        mapping = json.load(stream)["id_to_dirname"]
    allowed_classes = set(mapping.values())
    allowed = set(Path(train_list).read_text(encoding="utf-8").split())
    if not allowed:
        raise ValueError("Training image list is empty")
    matrices, labels, groups, records = [], [], [], []
    classes = []
    for directory in sorted((root / "features").iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.npz"))
        if not files:
            continue
        if len(files) != 1:
            raise ValueError(f"Expected one matrix in {directory}")
        x, ids = load_feature_matrix_npz(files[0])
        if x.shape[1] != signature["embed_dim"]:
            raise ValueError(f"Wrong feature dimension in {files[0]}")
        image_ids = np.array([image_id_from_sample(s) for s in ids])
        unknown = set(image_ids) - allowed
        if unknown:
            raise ValueError(f"Bank contains images outside --train-list: {sorted(unknown)[:5]}")
        valid = np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 1e-6)
        x, ids, image_ids = x[valid], ids[valid], image_ids[valid]
        if not len(x):
            continue
        if directory.name not in allowed_classes:
            raise ValueError(f"Class {directory.name} missing from source class_mapping.json")
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        labels.extend([len(classes)] * len(x))
        classes.append(directory.name)
        matrices.append(x)
        groups.extend(image_ids.tolist())
        records.append((files[0].relative_to(root), x, ids))
    if len(classes) < 2:
        raise ValueError("Need at least two nonempty classes")
    return np.vstack(matrices), np.array(labels), np.array(groups), classes, signature, records


def split_by_image(groups, labels, val_fraction, seed):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    images = np.unique(groups)
    if len(images) < 3:
        raise ValueError("Need at least three training images")
    rng = np.random.default_rng(seed)
    images = rng.permutation(images)
    count = min(len(images) - 1, max(1, round(len(images) * val_fraction)))
    validation = np.isin(groups, images[:count])
    train = np.flatnonzero(~validation)
    val = np.flatnonzero(validation)
    missing = set(labels) - set(labels[train])
    if missing:
        raise ValueError(f"Classes {sorted(missing)} absent from training partition; change split seed/fraction")
    return train, val


def split_from_report(groups, labels, report_path):
    """Keep baseline validation image membership after feature-space filtering.

    An image with no surviving baseline features can appear in a new bank. It
    joins training, never the frozen validation set. Actual coverage is reported.
    """
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    train_images, val_images = set(report["train_images"]), set(report["validation_images"])
    if not train_images or not val_images or train_images & val_images:
        raise ValueError("Invalid or overlapping reference train/validation images")
    validation = np.isin(groups, sorted(val_images))
    train, val = np.flatnonzero(~validation), np.flatnonzero(validation)
    if not len(val) or set(labels) - set(labels[train]):
        raise ValueError("Frozen split has no validation features or misses training classes")
    return train, val


def balanced_batch(labels, groups, train, rng, classes_per_batch, instances_per_class):
    classes = np.unique(labels[train])
    chosen = rng.choice(classes, min(classes_per_batch, len(classes)), replace=False)
    result = []
    for cls in chosen:
        rows = train[labels[train] == cls]
        image_ids = np.unique(groups[rows])
        # Distinct images supply positive pairs; do not duplicate one vector.
        selected_images = rng.permutation(image_ids)[:instances_per_class]
        result.extend(int(rng.choice(rows[groups[rows] == image])) for image in selected_images)
    return np.array(result, dtype=np.int64)


def retrieval_metrics(gallery, gallery_labels, queries, query_labels, k=5,
                      mode="knn", class_topm=5, centroid_weight=0.4,
                      search_multiplier=2):
    """Exact cosine kNN, chunked to avoid an all-pairs dataset matrix."""
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8)
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    k = min(k, len(gallery))
    if mode not in {"knn", "balanced"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    predictions = []
    if mode == "balanced":
        from src.retrieval import build_retrieval_context, classify_queries
        labels = [str(int(label)) for label in gallery_labels]
        context = build_retrieval_context(gallery, labels)
        for start in range(0, len(queries), 128):
            batch = classify_queries(
                queries[start:start + 128], gallery, labels, None, context,
                topk=k, mode="balanced", class_topm=class_topm,
                centroid_weight=centroid_weight, search_multiplier=search_multiplier)
            predictions.extend(int(item.label) for item in batch)
        return _classification_metrics(predictions, query_labels)
    for start in range(0, len(queries), 128):
        similarities = queries[start:start + 128] @ gallery.T
        indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        for row, neighbors in zip(similarities, indices):
            labs = gallery_labels[neighbors]
            ranked = sorted(np.unique(labs), key=lambda c: (
                int((labs == c).sum()), float(row[neighbors[labs == c]].sum()), -int(c)
            ), reverse=True)
            predictions.append(ranked[0])
    return _classification_metrics(predictions, query_labels)


def _classification_metrics(predictions, query_labels):
    predictions = np.array(predictions)
    recalls = [float((predictions[query_labels == c] == c).mean())
               for c in np.unique(query_labels)]
    return {"accuracy": float((predictions == query_labels).mean()),
            "macro_accuracy": float(np.mean(recalls)),
            "per_class_accuracy": {str(int(c)): float((predictions[query_labels == c] == c).mean())
                                   for c in np.unique(query_labels)},
            "evaluated_classes": int(len(recalls))}
