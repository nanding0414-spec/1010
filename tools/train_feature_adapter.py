#!/usr/bin/env python3
"""Train only an MLP on cached training-set SAM3 SMFA features, then export a bank."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.adapter_training import (load_training_bank, split_by_image, balanced_batch,
                                  retrieval_metrics, split_from_report)
from src.feature_adapter import (FeatureAdapter, supervised_contrastive_loss,
                                 project_features, load_adapter, adapted_signature,
                                 CosineClassifier, feature_consistency_loss)
from src.feature_bank_storage import save_feature_matrix_npz
from src.feature_signature import save_feature_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--train-list", required=True, help="Allowed TRAIN image stems, one per line")
    parser.add_argument("--output-dir", required=True, help="New directory for weights and metrics")
    parser.add_argument("--export-bank", required=True, help="New projected bank directory")
    parser.add_argument("--recipe", choices=["v1", "v2"], default="v1",
                        help="v2: cosine classifier, regularization, balanced checkpoint selection")
    parser.add_argument("--classifier", choices=["linear", "cosine"], default=None)
    parser.add_argument("--selection-metric", choices=["knn", "balanced"], default=None)
    parser.add_argument("--feature-dropout", type=float, default=None)
    parser.add_argument("--consistency-weight", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--cosine-scale", type=float, default=16.0)
    parser.add_argument("--eval-class-topm", type=int, default=5)
    parser.add_argument("--eval-centroid-weight", type=float, default=0.4)
    parser.add_argument("--eval-search-multiplier", type=int, default=2)
    parser.add_argument("--eval-topk", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--instances-per-class", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, choices=[2, 3], default=3,
                        help="Number of Linear layers in the adapter (default: 3)")
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path,
                        help="Reuse validation image membership from a baseline training_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    from configs.experiment import TRAINING
    parser.set_defaults(**TRAINING)
    args = parser.parse_args()
    defaults = (dict(classifier="cosine", selection_metric="balanced", feature_dropout=0.1,
                     consistency_weight=0.2, label_smoothing=0.05, eval_topk=25)
                if args.recipe == "v2" else
                dict(classifier="linear", selection_metric="knn", feature_dropout=0.0,
                     consistency_weight=0.0, label_smoothing=0.0, eval_topk=5))
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.epochs, args.steps_per_epoch, args.hidden_dim, args.output_dim) < 1:
        parser.error("epochs, steps and dimensions must be positive")
    if args.classes_per_batch < 2 or args.instances_per_class < 2:
        parser.error("Use at least two classes and two instances per class")
    if args.lr <= 0 or args.temperature <= 0 or args.contrastive_weight < 0:
        parser.error("Invalid learning rate, temperature or contrastive weight")
    if not 0 <= args.feature_dropout < 1 or not 0 <= args.label_smoothing < 1:
        parser.error("feature-dropout and label-smoothing must be in [0, 1)")
    if args.consistency_weight < 0 or args.cosine_scale <= 0:
        parser.error("Invalid consistency weight or cosine scale")
    if min(args.eval_class_topm, args.eval_search_multiplier, args.eval_topk) < 1:
        parser.error("Evaluation neighbor counts must be positive")
    if not 0 <= args.eval_centroid_weight <= 1:
        parser.error("eval-centroid-weight must be in [0, 1]")
    out = Path(args.output_dir).resolve()
    bank_out = Path(args.export_bank).resolve()
    source = Path(args.source_bank).resolve()
    for target in (out, bank_out):
        if target.exists() or target == source or source in target.parents or target in source.parents:
            parser.error(f"Choose a fresh directory outside the source bank: {target}")
    if out == bank_out or out in bank_out.parents or bank_out in out.parents:
        parser.error("Weights and exported bank must be separate directories")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y, groups, classes, signature, records = load_training_bank(source, args.train_list)
    if args.split_file:
        args.split_file = str(args.split_file.resolve())
    train, val = (split_from_report(groups, y, args.split_file) if args.split_file
                  else split_by_image(groups, y, args.val_fraction, args.seed))
    bg_index = classes.index("background") if "background" in classes else None
    if not any(len(np.unique(groups[train[y[train] == cls]])) >= 2
               for cls in np.unique(y) if cls != bg_index):
        raise ValueError("No foreground class has two distinct training images for positive pairs")
    out.mkdir(parents=True)
    device = torch.device(args.device)
    model = FeatureAdapter(x.shape[1], args.hidden_dim, args.output_dim,
                           num_layers=args.num_layers).to(device)
    print(f"[Adapter] {model.architecture}")
    classifier = (CosineClassifier(args.output_dim, len(classes), args.cosine_scale)
                  if args.classifier == "cosine" else nn.Linear(args.output_dim, len(classes))).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    baseline = retrieval_metrics(x[train], y[train], x[val], y[val])
    evaluation = dict(k=args.eval_topk, mode=args.selection_metric,
                      class_topm=args.eval_class_topm, centroid_weight=args.eval_centroid_weight,
                      search_multiplier=args.eval_search_multiplier)
    selection_baseline = retrieval_metrics(x[train], y[train], x[val], y[val], **evaluation)
    print(f"[Data] {len(train)} train / {len(val)} held-out instances; {len(classes)} classes")
    print(f"[Raw SAM3 kNN] {baseline}")
    print(f"[Recipe] {args.recipe}, classifier={args.classifier}, selection={evaluation}")
    print(f"[Raw SAM3 selection baseline] {selection_baseline}")
    print("[Validation] Held-out TRAIN images, pseudo labels; these scores are not GT mIoU.")
    report = {"args": vars(args), "classes": classes, "baseline": baseline,
              "selection_baseline": selection_baseline,
              "train_images": sorted(set(groups[train].tolist())),
              "validation_images": sorted(set(groups[val].tolist())), "epochs": []}
    checkpoint = out / "adapter.pt"
    best = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()
        losses = []
        components = []
        for _ in range(args.steps_per_epoch):
            rows = balanced_batch(y, groups, train, rng, args.classes_per_batch,
                                  args.instances_per_class)
            batch_x = torch.from_numpy(x[rows]).to(device)
            batch_y = torch.from_numpy(y[rows]).long().to(device)
            z = model(batch_x)
            ce = F.cross_entropy(classifier(z), batch_y, label_smoothing=args.label_smoothing)
            contrastive = supervised_contrastive_loss(z, batch_y, args.temperature, bg_index)
            consistency = (feature_consistency_loss(model, batch_x, z, args.feature_dropout)
                           if args.consistency_weight > 0 else z.sum() * 0)
            loss = ce + args.contrastive_weight * contrastive + args.consistency_weight * consistency
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite adapter training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append([float(ce.detach()), float(contrastive.detach()), float(consistency.detach())])
        scheduler.step()
        model.eval()
        z = project_features(model, x, device)
        metrics = retrieval_metrics(z[train], y[train], z[val], y[val], **evaluation)
        mean_components = np.mean(components, axis=0)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics,
               "ce_loss": float(mean_components[0]), "contrastive_loss": float(mean_components[1]),
               "consistency_loss": float(mean_components[2])}
        report["epochs"].append(row)
        print(f"[Epoch {epoch:03d}] loss={row['loss']:.4f} {args.selection_metric} macro={metrics['macro_accuracy']:.4f}")
        if metrics["macro_accuracy"] > best:
            best, best_epoch = metrics["macro_accuracy"], epoch
            payload = {"format_version": 1, "architecture": model.architecture,
                       "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "source_signature": signature, "classes": classes,
                       "epoch": epoch, "validation": metrics, "train_args": vars(args)}
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint)
    report.update(best_epoch=best_epoch, best_macro_accuracy=best)
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best <= selection_baseline["macro_accuracy"]:
        print("[Warning] Adapter did not beat raw-feature held-out retrieval. Keep the raw baseline for comparison.")

    model, _ = load_adapter(checkpoint, device, signature)
    bank_out.mkdir(parents=True)
    (bank_out / "meta").mkdir()
    shutil.copy2(source / "meta" / "class_mapping.json", bank_out / "meta" / "class_mapping.json")
    stats = {}
    for relative_path, features, ids in records:
        projected = project_features(model, features, device)
        save_feature_matrix_npz(bank_out / relative_path, projected, ids)
        stats[relative_path.parent.name] = {"count": len(projected)}
    save_feature_signature(bank_out / "meta", adapted_signature(signature, checkpoint, model.architecture))
    (bank_out / "meta" / "adapter_export.json").write_text(json.dumps({
        "source_bank": str(source), "adapter_checkpoint": str(checkpoint),
        "best_epoch": best_epoch, "counts": stats,
        "note": "Includes all source TRAIN instances after internal validation; no additional filtering."
    }, indent=2), encoding="utf-8")
    print(f"[Done] Adapter: {checkpoint}")
    print(f"[Done] Projected bank: {bank_out}; inference builds its FAISS index from these matrices.")


if __name__ == "__main__":
    main()
