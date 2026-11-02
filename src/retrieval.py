"""Feature-bank retrieval and class-balanced scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def canonical_label(label: str) -> str:
    label = str(label)
    return "background" if label == "background" or label.startswith("background_") else label


@dataclass(frozen=True)
class RetrievalContext:
    labels: tuple[str, ...]
    centroids: np.ndarray
    counts: np.ndarray
    prototypes: tuple[np.ndarray, ...] | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class RetrievalPrediction:
    label: str
    score: float
    margin: float
    class_scores: tuple[float, ...] | None = None


def build_retrieval_context(
    all_feats: np.ndarray,
    all_labels: Sequence[str],
    prototype_count: int = 1,
    sample_weights=None,
    temperature=None,
) -> RetrievalContext:
    feats = np.asarray(all_feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [N, C] gallery, got {feats.shape}")
    if len(all_labels) != feats.shape[0]:
        raise ValueError("Gallery labels must match gallery feature rows")

    labels_array = np.asarray([canonical_label(x) for x in all_labels], dtype=str)
    labels = tuple(dict.fromkeys(labels_array.tolist()))
    centroids = []
    counts = []
    prototypes = []
    if prototype_count < 1:
        raise ValueError('Prototype count must be positive')
    weights = np.ones(len(feats), np.float32) if sample_weights is None else np.asarray(sample_weights, np.float32)
    if weights.shape != (len(feats),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid gallery member weights')
    for label in labels:
        cls_feats = feats[labels_array == label]
        center = (cls_feats.mean(axis=0) if sample_weights is None else
                  np.average(cls_feats, axis=0, weights=weights[labels_array == label]))
        center /= np.linalg.norm(center) + 1e-8
        centroids.append(center)
        counts.append(cls_feats.shape[0])
        if prototype_count > 1:
            from src.optimization import spherical_centers
            prototypes.append(spherical_centers(cls_feats, prototype_count, weights[labels_array == label]))
    return RetrievalContext(
        labels=labels,
        centroids=np.ascontiguousarray(np.vstack(centroids).astype(np.float32)),
        counts=np.asarray(counts, dtype=np.int64),
        prototypes=tuple(prototypes) if prototype_count > 1 else None,
        temperature=temperature,
    )


def _search(query_feats, all_feats, faiss_index, k):
    if faiss_index is not None:
        return faiss_index.search(query_feats, k)

    sims_all = query_feats @ all_feats.T
    idxs_batch = np.argpartition(sims_all, -k, axis=1)[:, -k:]
    row_ids = np.arange(sims_all.shape[0])[:, None]
    order = np.argsort(sims_all[row_ids, idxs_batch], axis=1)[:, ::-1]
    idxs_batch = idxs_batch[row_ids, order]
    return sims_all[row_ids, idxs_batch], idxs_batch


def classify_queries(
    query_feats: np.ndarray,
    all_feats: np.ndarray,
    all_labels: Sequence[str],
    faiss_index,
    context: RetrievalContext,
    topk: int,
    mode: str = "balanced",
    class_topm: int = 5,
    search_multiplier: int = 2,
    centroid_weight: float = 0.4,
    class_topm_by_label: dict | None = None,
    centroid_fallback: bool = True,
) -> List[RetrievalPrediction]:
    """Classify normalized queries with legacy voting or balanced class scores.

    Balanced mode retrieves a wider candidate set, averages only each class's
    strongest M neighbors, and blends that local score with an equal-prior
    class centroid score. This reduces the advantage of classes that simply
    contain more gallery samples.
    """
    query_feats = np.ascontiguousarray(np.asarray(query_feats, dtype=np.float32))
    all_feats = np.ascontiguousarray(np.asarray(all_feats, dtype=np.float32))
    mode = str(mode).lower()
    if mode not in {"legacy", "balanced"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if not 0.0 <= centroid_weight <= 1.0:
        raise ValueError("centroid_weight must be in [0, 1]")

    n_gallery = all_feats.shape[0]
    if class_topm_by_label is not None and any(type(v) is not int or v < 1 for v in class_topm_by_label.values()):
        raise ValueError('Invalid per-class topm')
    largest_topm = max([class_topm, *(class_topm_by_label or {}).values()])
    if mode == "legacy":
        search_k = min(max(int(topk), 1), n_gallery)
    else:
        search_k = min(
            max(
                int(topk),
                max(int(largest_topm), 1)
                * len(context.labels)
                * max(int(search_multiplier), 1),
            ),
            n_gallery,
        )

    sims_batch, idxs_batch = _search(query_feats, all_feats, faiss_index, search_k)
    gallery_labels = np.asarray([canonical_label(x) for x in all_labels], dtype=str)
    centroid_scores = query_feats @ context.centroids.T
    if context.prototypes is not None:
        centroid_scores = np.column_stack([(query_feats @ p.T).max(1) for p in context.prototypes])
    predictions = []

    for query_idx, (sims, idxs) in enumerate(zip(sims_batch, idxs_batch)):
        saved_scores = None
        valid = np.asarray(idxs) >= 0
        sims = np.asarray(sims, dtype=np.float32)[valid]
        idxs = np.asarray(idxs, dtype=np.int64)[valid]

        if mode == "legacy":
            votes = {}
            sums = {}
            for sim, gallery_idx in zip(sims, idxs):
                label = gallery_labels[gallery_idx]
                votes[label] = votes.get(label, 0) + 1
                sums[label] = sums.get(label, 0.0) + float(sim)
            ranked = sorted(
                votes,
                key=lambda label: (votes[label], sums[label]),
                reverse=True,
            )
            best = ranked[0]
            best_score = sums[best] / votes[best]
            second_score = (
                sums[ranked[1]] / votes[ranked[1]] if len(ranked) > 1 else -1.0
            )
        else:
            neighbor_labels = gallery_labels[idxs]
            class_scores = np.empty(len(context.labels), dtype=np.float32)
            for class_idx, label in enumerate(context.labels):
                cls_sims = sims[neighbor_labels == label]
                if cls_sims.size:
                    requested = (class_topm_by_label or {}).get(label, class_topm)
                    topm = min(max(int(requested), 1), cls_sims.size)
                    local_score = float(np.partition(cls_sims, -topm)[-topm:].mean())
                else:
                    local_score = float(centroid_scores[query_idx, class_idx]) if centroid_fallback else -1.0
                class_scores[class_idx] = (
                    (1.0 - centroid_weight) * local_score
                    + centroid_weight * float(centroid_scores[query_idx, class_idx])
                )
            ranking = np.argsort(class_scores)[::-1]
            saved_scores = tuple(map(float, class_scores))
            if context.temperature is not None:
                from src.optimization import probabilities
                class_scores = probabilities(class_scores[None,:], context.temperature)[0]
            best_idx = int(ranking[0])
            best = context.labels[best_idx]
            best_score = float(class_scores[best_idx])
            second_score = float(class_scores[ranking[1]]) if len(ranking) > 1 else -1.0

        predictions.append(
            RetrievalPrediction(
                label=str(best),
                score=float(best_score),
                margin=float(best_score - second_score),
                class_scores=saved_scores,
            )
        )
    return predictions


__all__ = [
    "RetrievalContext",
    "RetrievalPrediction",
    "build_retrieval_context",
    "canonical_label",
    "classify_queries",
]
