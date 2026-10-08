#!/usr/bin/env python3
"""Compare proposal cleanup settings by validation mIoU using a fixed SAM3/MLP bank."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.proposal_cleanup import CleanupConfig

RETRIEVAL_BASE = {
    "INF_CENTROID_WEIGHT": 0.4, "INF_CLASS_TOPM": 5,
    "INF_PROPOSAL_SCORE_WEIGHT": 0.2, "INF_RETRIEVAL_MARGIN_THR": 0.0,
}


def retrieval_cases(min_score=0.7):
    cleanup = CleanupConfig(min_score=min_score).to_dict()
    cases = []
    def add(name, settings=None, nms=0.85):
        cases.append(dict(name=name, cleanup={**cleanup, "nms_iou": nms},
                          retrieval={**RETRIEVAL_BASE, **(settings or {})}))
    add("baseline_mlp3")
    for weight in (0.0, 0.2, 0.4, 0.6, 0.8):
        for topm in (1, 3, 5, 10):
            if weight == 0.4 and topm == 5:
                continue
            add(f"centroid_{weight}_topm_{topm}",
                {"INF_CENTROID_WEIGHT": weight, "INF_CLASS_TOPM": topm})
    # Single-factor controls around the same baseline; no hidden adaptive choices.
    for value in (0.0, 0.4, 0.6):
        add(f"proposal_weight_{value}", {"INF_PROPOSAL_SCORE_WEIGHT": value})
    for value in (0.02, 0.05, 0.1):
        add(f"margin_{value}", {"INF_RETRIEVAL_MARGIN_THR": value})
    for value in (0.7, 0.95, 1.0):
        add(f"cleanup_nms_{value}", nms=value)
    return cases


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def default_cases():
    base = CleanupConfig()
    cases = [{"name": "baseline_raw", "cleanup": None},
             {"name": "cleanup_default", "cleanup": base.to_dict()}]
    variants = {
        "min_score": [0.0, 0.05, 0.10, 0.20],
        "min_island_area": [0, 32, 128],
        "min_mask_area": [150, 300],
        "max_hole_area": [0, 64],
        "nms_iou": [0.70, 1.0],
    }
    for key, values in variants.items():
        for value in values:
            cases.append({"name": f"{key}_{value}", "cleanup": replace(base, **{key: value}).to_dict()})
    cases.append({"name": "conservative_combined", "cleanup": replace(
        base, min_score=0.05, min_island_area=32, max_hole_area=16, nms_iou=0.95,
    ).to_dict()})
    return cases


def load_cases(path=None):
    cases = json.loads(Path(path).read_text(encoding="utf-8")) if path else default_cases()
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases JSON must be a nonempty list of {name, cleanup} objects")
    if not any(case.get("cleanup") is None for case in cases):
        cases.insert(0, {"name": "baseline_raw", "cleanup": None})
    names = set()
    for case in cases:
        name = case["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in names:
            raise ValueError(f"Unsafe or duplicate case name: {name}")
        names.add(name)
        if case["cleanup"] is not None:
            case["cleanup"] = CleanupConfig(**case["cleanup"]).to_dict()
    # Always establish the unchanged-proposal baseline first.
    return sorted(cases, key=lambda case: case["cleanup"] is not None)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths):
    return [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(paths)]


def make_env(args, signature):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.cuda), PYTHONUNBUFFERED="1",
        OVERRIDE_DATASET_TYPE="voc", OVERRIDE_FEATURE_BACKBONE="sam3",
        OVERRIDE_FEATURE_ADAPTER_CKPT=str(args.adapter),
        OVERRIDE_INF_FEATURE_BANK_ROOT=str(args.bank),
        OVERRIDE_INF_PROPOSALS_DIR=str(args.proposals_dir),
        OVERRIDE_INF_LOAD_PROPOSALS="true", OVERRIDE_INF_ALLOW_LEGACY_FEATURE_BANK="false",
        OVERRIDE_INF_VIS_LIMIT="0", OVERRIDE_MIN_MASK_AREA=str(args.inference_min_mask_area),
    )
    for field, variable in {
        "feature_source": "OVERRIDE_SAM3_FEATURE_SOURCE", "feature_level": "OVERRIDE_SAM3_FEATURE_LEVEL",
        "resolution": "OVERRIDE_SAM3_RESOLUTION", "token_norm": "OVERRIDE_SAM3_TOKEN_NORM",
        "context_subtract": "OVERRIDE_SAM3_CONTEXT_SUBTRACT", "use_bf16": "OVERRIDE_SAM3_USE_BF16",
    }.items():
        value = signature[field]
        env[variable] = str(value).lower() if isinstance(value, bool) else str(value)
    if signature.get('pooling') or signature.get('aggregation'):
        raise ValueError('Only single-bank SMFA signatures are supported')
    return env


def read_result(path, case, reference=None):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not math.isfinite(result["mIoU"]) or not 0 <= result["mIoU"] <= 1:
        raise ValueError("Invalid mIoU")
    if result["cleanup"] != case["cleanup"]:
        raise ValueError("Evaluated cleanup parameters do not match the requested case")
    variable_keys = set(case.get("retrieval", {}))
    for key, value in case.get("retrieval", {}).items():
        if result["inference_settings"].get(key) != value:
            raise ValueError(f"Evaluated retrieval setting differs: {key}")
    ids = result["evaluated_ids"]
    if not ids or len(set(ids)) != len(ids) or len(ids) != result["evaluated_count"]:
        raise ValueError("Invalid evaluation coverage")
    if reference:
        for field in ("evaluated_ids", "feature_signature", "inference_settings", "feature_bank", "proposals_dir"):
            actual, expected = result[field], reference[field]
            if field == "inference_settings":
                actual = {k: v for k, v in actual.items() if k not in variable_keys}
                expected = {k: v for k, v in expected.items() if k not in variable_keys}
            if actual != expected:
                raise ValueError(f"Trials differ in {field}; refusing to rank incomparable results")
    return result


def save_ranking(output, completed, source):
    ranked = sorted(completed, key=lambda row: -row["mIoU"])
    baseline = next((row["mIoU"] for row in completed if row["cleanup"] is None), completed[0]["mIoU"])
    fields = ["rank", "name", "mIoU", "mIoU_percent", "delta_percentage_points", "evaluated_count",
              "proposal_count_before", "proposal_count_after", *CleanupConfig().to_dict(),
              *RETRIEVAL_BASE, "metrics_json"]
    temporary = output / "leaderboard.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow(dict(
                rank=rank, name=row["name"], mIoU=row["mIoU"], mIoU_percent=100 * row["mIoU"],
                delta_percentage_points=100 * (row["mIoU"] - baseline),
                evaluated_count=row["evaluated_count"], proposal_count_before=row["proposal_count_before"],
                proposal_count_after=row["proposal_count_after"], metrics_json=row["metrics_json"],
                **(row["cleanup"] or {}),
                **row.get("retrieval", {}),
            ))
    temporary.replace(output / "leaderboard.csv")
    best = dict(ranked[0])
    best["completed_trials"] = len(completed)
    best["selection_note"] = "Best among tested settings on this validation set; not an independent test score."
    if best["cleanup"] is None:
        best["export_command"] = None
        best["proposal_directory"] = str(source)
        write_json(output / "best_cleanup_config.json", None)
    else:
        write_json(output / "best_cleanup_config.json", best["cleanup"])
        command = ["python", "tools/refine_proposals.py", "--input-dir", str(source),
                   "--output-dir", str(output / "best_proposals")]
        for key, value in best["cleanup"].items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        best["export_command"] = shlex.join(command)
    write_json(output / "best.json", best)
    if best.get("retrieval"):
        write_json(output / "best_retrieval_overrides.json",
                   {"OVERRIDE_" + key: value for key, value in best["retrieval"].items()})
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals-dir", type=Path, required=True, help="Original, not previously cleaned, proposal cache")
    parser.add_argument("--bank", type=Path, required=True, help="MLP-exported feature bank")
    parser.add_argument("--adapter", type=Path, required=True, help="Matching adapter.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda", default="0", help="One GPU ID; trials run sequentially")
    parser.add_argument("--cases", type=Path, help="Custom JSON list; default: baseline + 15 cleanup settings")
    parser.add_argument("--retrieval-sweep", action="store_true",
                        help="29 trials: centroid/Top-M grid and score/margin/NMS controls")
    parser.add_argument("--min-score", type=float, default=0.7,
                        help="Fixed cleanup confidence for --retrieval-sweep")
    parser.add_argument("--eval-limit", type=int, default=0, help="Diagnostic subset; default 0 evaluates all VOC val images")
    parser.add_argument("--inference-min-mask-area", type=int, default=100, help="Fixed downstream mask-area floor for every trial")
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials only if inputs/settings are unchanged")
    args = parser.parse_args(argv)
    if args.eval_limit < 0 or args.inference_min_mask_area < 0:
        parser.error("Limits must be nonnegative")
    for name in ("proposals_dir", "bank", "adapter", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.retrieval_sweep and args.cases:
        parser.error("--retrieval-sweep and --cases are mutually exclusive")
    cases = retrieval_cases(args.min_score) if args.retrieval_sweep else load_cases(args.cases)
    if not args.proposals_dir.is_dir() or not args.adapter.is_file():
        parser.error("Proposal directory and adapter checkpoint must exist")
    signature_path = args.bank / "meta/feature_signature.json"
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if signature.get("backbone") != "sam3" or not signature.get("adapter"):
        parser.error("--bank must be an MLP-adapted SAM3 bank")
    if signature["adapter"]["sha256"] != sha256(args.adapter):
        parser.error("Adapter checkpoint does not match the exported bank")
    if args.retrieval_sweep and signature["adapter"]["architecture"].get("num_layers", 2) != 3:
        parser.error("--retrieval-sweep expects the three-layer MLP bank")
    for source in (args.proposals_dir, args.bank, args.adapter.parent):
        if args.output_dir == source or source in args.output_dir.parents or args.output_dir in source.parents:
            parser.error("Sweep output must be separate from proposal, bank and adapter directories")
    proposal_files = sorted(args.proposals_dir.glob("*.npz"))
    bank_files = sorted((args.bank / "features").glob("*/*.npz"))
    if not proposal_files or not bank_files:
        parser.error("Proposal cache or feature bank is empty")
    env = make_env(args, signature)
    if args.retrieval_sweep:
        env.update({"OVERRIDE_" + key: str(value) for key, value in {
            **RETRIEVAL_BASE, "INF_RETRIEVAL_MODE": "balanced", "INF_TOPK_NEIGH": 25,
            "INF_RETRIEVAL_SEARCH_MULTIPLIER": 2, "INF_NMS_IOU": 0.4,
            "INF_MASK_BACKEND": "sam3_query", "INF_FAISS_DEVICE": 0, "SEED": 42,
        }.items()})
    # File sizes/mtimes detect ordinary changes on resume; checkpoints also have a SHA256 check.
    data_paths = [ROOT / "data/VOC2012/ImageSets/Segmentation/val.txt"]
    for directory in ("JPEGImages", "SegmentationClassAug", "SegmentationClass"):
        data_paths.extend((ROOT / "data/VOC2012" / directory).glob("*"))
    source_paths = proposal_files + bank_files + [signature_path, args.adapter, args.bank / "meta/class_mapping.json"]
    source_paths.extend(path for path in data_paths if path.is_file())
    checkpoint = Path(env.get("OVERRIDE_SAM3_CKPT") or "weights/sam3/sam3.pt").expanduser()
    checkpoint = (checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"Local SAM3 checkpoint required: {checkpoint}")
    source_paths.append(checkpoint)
    code_paths = [ROOT / "run_inference.py", Path(__file__).resolve()]
    for directory in ("src", "configs", "project_utils"):
        code_paths.extend((ROOT / directory).glob("*.py"))
    manifest = {
        "cases": cases, "eval_limit": args.eval_limit,
        "environment": {key: value for key, value in env.items() if key.startswith("OVERRIDE_")},
        "inputs": snapshot(source_paths),
        "code": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
    }
    manifest_path = args.output_dir / "sweep_manifest.json"
    if args.output_dir.exists():
        if not args.resume or not manifest_path.is_file():
            parser.error("Choose a new output directory, or use --resume for a matching sweep")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            parser.error("Inputs/settings/code changed; use a new output directory")
    else:
        args.output_dir.mkdir(parents=True)
        write_json(manifest_path, manifest)
    write_json(args.output_dir / "cases.json", cases)
    completed, reference = [], None
    print(f"[Sweep] {len(cases)} trials; GPU={args.cuda}; eval_limit={args.eval_limit} (0=full val)", flush=True)
    print("[Sweep] Masks are cleaned in memory. Banks, weights and raw proposal files are not changed.", flush=True)
    for number, case in enumerate(cases, 1):
        trial = args.output_dir / case["name"]
        trial.mkdir(exist_ok=True)
        result_path = trial / "metrics.json"
        if not result_path.exists():
            trial_env = dict(env)
            trial_env.update({"OVERRIDE_" + key: str(value)
                              for key, value in case.get("retrieval", {}).items()})
            command = [sys.executable, "-u", str(ROOT / "run_inference.py"), "--strict-eval",
                       "--output-dir", str(trial), "--results-json", str(result_path),
                       "--eval-limit", str(args.eval_limit)]
            if case["cleanup"] is not None:
                config_path = trial / "cleanup.json"
                write_json(config_path, case["cleanup"])
                command.extend(["--cleanup-config", str(config_path)])
            print(f"[{number}/{len(cases)}] {case['name']} -> {trial / 'inference.log'}", flush=True)
            started = time.monotonic()
            with (trial / "inference.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, env=trial_env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f"  Elapsed: {(time.monotonic() - started) / 60:.1f} min", flush=True)
        result = read_result(result_path, case, reference)
        if result["feature_signature"] != signature:
            raise ValueError("Inference feature signature differs from the selected bank")
        reference = reference or result
        completed.append(dict(result, name=case["name"], metrics_json=str(result_path),
                              retrieval=case.get("retrieval", {})))
        best = save_ranking(args.output_dir, completed, args.proposals_dir)
        print(f"  mIoU={result['mIoU']:.6f}; current best={best['name']} ({100 * best['mIoU']:.2f}%)", flush=True)
    if snapshot(source_paths) != manifest["inputs"]:
        raise RuntimeError("Inputs changed during the sweep; results must not be treated as a controlled comparison")
    write_json(args.output_dir / "sweep_complete.json", {"trials": len(completed), "best": best["name"]})
    print(f"[DONE] Best: {best['name']}, mIoU={100 * best['mIoU']:.2f}%", flush=True)
    print(f"[Ranking] {args.output_dir / 'leaderboard.csv'}", flush=True)
    if best["export_command"]:
        print(f"[Export best proposals]\n{best['export_command']}", flush=True)
    else:
        print("[Best] No cleanup improved over the baseline; reuse the original proposals.", flush=True)


if __name__ == "__main__":
    main()
