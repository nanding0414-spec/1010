"""Run the fixed R3 record recipe through the original, unchanged worker.

This is a staged inference entrypoint, not a training or evaluation launcher.
See PROJECT_STORAGE_MANUAL section 84 for the signed manifest/job contracts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import oracle_guided_r as worker

PROFILE = ROOT / "configs/recipes/r3_record_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate(profile, manifest, job):
    """Reject a different recipe or missing ownership before any model load."""
    if (profile["recipe_id"] != "r3_record_v1" or profile["arm"] != "R3"
            or profile["worker_rules"] != worker.RULES
            or profile["crop_fraction"] != .8):
        raise ValueError("This entrypoint only supports the fixed R3 record recipe")
    if worker.sha256(ROOT / "tools/oracle_guided_r.py") != profile["worker_sha256"]:
        raise ValueError("R3 worker SHA differs from the published recipe")
    for key in ("version", "source_signature", "sam3_weights", "source_adapter",
                "source_mlp_bank", "token_records", "code_digest", "parameter_digest"):
        if key not in manifest or not manifest[key]:
            raise ValueError(f"Missing signed manifest field: {key}")
    if not manifest["version"].strip() or manifest["version"] in {".", ".."}:
        raise ValueError("A new explicit run version is required")
    signature = manifest["source_signature"]
    if signature != profile["source_signature"]:
        raise ValueError("R3 requires the original FP32 1008 hard-center GMP signature")
    if manifest.get("recipe_id") != profile["recipe_id"] or manifest.get("baseline_recipe") != profile["baseline"]:
        raise ValueError("Freeze the matching center=.2 GMP+HCR baseline recipe")
    if manifest.get("seed") not in profile["record"]["seeds"]:
        raise ValueError("The record recipe supports the three original MLP seeds only")
    if job.get("arm") != "R3" or job.get("stage") not in {"context", "generate", "query", "rescue"}:
        raise ValueError("Only R3 context/generate/query/rescue jobs are supported")
    stage = job["stage"]
    if not job.get("context_path") or not job.get("output_dir"):
        raise ValueError("Explicit context_path and output_dir are required")
    image_ids = job.get("image_ids", [])
    if stage != "context" and (not image_ids or len(image_ids) != len(set(image_ids))):
        raise ValueError("Use an explicit nonempty, unique image ID list")
    for image_id in image_ids:
        if not isinstance(image_id, str) or not image_id or Path(image_id).name != image_id or "/" in image_id or "\\" in image_id or image_id in {".", ".."}:
            raise ValueError("Image IDs must be plain file stems")
        if image_id not in manifest["token_records"]:
            raise ValueError(f"Missing original token record: {image_id}")
        record = manifest["token_records"][image_id]
        for key in ("path", "sha256", "receipt_path", "receipt_sha256", "geometry"):
            if key not in record:
                raise ValueError(f"Missing token {key}: {image_id}")
        if record["geometry"].get("token_hw") != [72, 72]:
            raise ValueError("R3 uses the original 72x72 token grid")
    # Freeze all resources loaded by the worker, including every class-bank file.
    records = manifest.get("input_artifacts", [])
    indexed = {str(Path(r["path"]).resolve()): r for r in records}
    bank = Path(manifest["source_mlp_bank"]).resolve()
    resources = [Path(manifest["sam3_weights"]).resolve(),
                 Path(manifest["source_adapter"]).resolve()]
    if not bank.is_dir() or not (bank / "features").is_dir() or not (bank / "meta").is_dir():
        raise ValueError("Missing original MLP bank features/meta directories")
    resources += sorted(p for sub in ("features", "meta") for p in (bank / sub).rglob("*") if p.is_file())
    for path in resources:
        if not path.is_file() or str(path) not in indexed:
            raise ValueError(f"Missing signed input artifact: {path}")
        # No token pre-scan: token contents are verified by the worker on consumption.
        record = indexed[str(path)]
        if worker.sha256(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
            raise ValueError(f"Input SHA/size mismatch: {path}")
    sam3_path, adapter_path = resources[:2]
    if indexed[str(sam3_path)]["sha256"] != signature["checkpoint_sha256"]:
        raise ValueError("SAM3 checkpoint differs from the source signature")
    bank_signature = read_json(bank / "meta/feature_signature.json")
    adapter = bank_signature.get("adapter", {})
    architecture = adapter.get("architecture", {})
    original = {k: v for k, v in bank_signature.items() if k != "adapter"}
    original["embed_dim"] = 1280
    if (original != signature or adapter.get("sha256") != indexed[str(adapter_path)]["sha256"]
            or architecture.get("num_layers", 2) != 2
            or tuple(architecture.get(k) for k in ("input_dim", "hidden_dim", "output_dim")) != (1280, 512, 256)):
        raise ValueError("Original GMP bank, MLP architecture or checkpoint binding differs")
    for image_id in image_ids:
        for field in ("image_records", "baseline_records"):
            if image_id not in manifest.get(field, {}):
                raise ValueError(f"Missing frozen {field}: {image_id}")
    output = Path(job["output_dir"]).resolve()
    protected = [bank, *(p.parent for p in resources[:2])]
    for field in ("baseline_dir", "candidates_dir", "queries_dir", "image_dir"):
        if job.get(field):
            protected.append(Path(job[field]).resolve())
    for image_id in image_ids:
        for field in ("image_records", "baseline_records", "token_records"):
            protected.append(Path(manifest[field][image_id]["path"]).resolve().parent)
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from all source directories")
    for field in ({"query": ["candidates_dir"], "rescue": ["candidates_dir", "queries_dir", "baseline_dir"]}.get(stage, [])):
        if not job.get(field):
            raise ValueError(f"Missing job field: {field}")
    if stage in {"query", "rescue"}:
        for field in (["candidates_dir"] if stage == "query" else ["candidates_dir", "queries_dir"]):
            for image_id in image_ids:
                marker = read_json(Path(job[field]) / image_id / "complete.json")
                binding = marker["binding"]
                for key, expected in dict(arm="R3", version=manifest["version"], image_id=image_id,
                                         code_digest=manifest["code_digest"],
                                         parameter_digest=manifest["parameter_digest"],
                                         source_signature=signature, rules=worker.RULES).items():
                    if binding.get(key) != expected:
                        raise ValueError(f"Upstream {key} binding differs: {image_id}")
                for record in marker["artifacts"]:
                    worker._read_verified(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Frozen run/input manifest JSON")
    parser.add_argument("--job", required=True, help="One R3 worker stage JSON")
    parser.add_argument("--profile", default=str(PROFILE))
    parser.add_argument("--gpu", type=int, default=1, help="Physical GPU outside Slurm; allocated visibility is preserved")
    parser.add_argument("--preflight-only", action="store_true", help="Verify recipe/input SHA only; no model or GT load")
    args = parser.parse_args(argv)
    profile, manifest, job = read_json(args.profile), read_json(args.manifest), read_json(args.job)
    validate(profile, manifest, job)
    if args.preflight_only:
        print(json.dumps(dict(status="preflight_passed", recipe=profile["recipe_id"],
                              version=manifest["version"], stage=job["stage"],
                              model_loaded=False, GT_read=False)))
        return 0
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if "CUDA_VISIBLE_DEVICES" not in os.environ and "SLURM_JOB_ID" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    job = dict(job, device="cuda:0")
    worker.run_worker(manifest, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
