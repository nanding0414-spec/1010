#!/usr/bin/env python3
"""Create a separate, cleaned proposal cache from existing SAM3 .npz files.

No model, feature bank, or GPU is required. Output masks/scores are compatible
with run_inference.py. The input directory is never modified.
"""

import argparse
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.proposal_cleanup import CleanupConfig, clean_proposals, proposal_stats


def save_atomic(path, **arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    defaults = CleanupConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must be a new directory, to avoid mixing settings")
    parser.add_argument("--min-score", type=float, default=defaults.min_score)
    parser.add_argument("--min-mask-area", type=int, default=defaults.min_mask_area)
    parser.add_argument("--min-island-area", type=int, default=defaults.min_island_area)
    parser.add_argument("--max-hole-area", type=int, default=defaults.max_hole_area)
    parser.add_argument("--max-hole-fraction", type=float, default=defaults.max_hole_fraction)
    parser.add_argument("--nms-iou", type=float, default=defaults.nms_iou)
    args = parser.parse_args(argv)
    try:
        config = CleanupConfig(**{key: getattr(args, key) for key in defaults.to_dict()})
    except ValueError as exc:
        parser.error(str(exc))
    source, target = args.input_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f"Input directory not found: {source}")
    if target == source or target.exists():
        parser.error("Output directory must be new; existing caches will not be overwritten")
    files = sorted(source.glob("*.npz"))
    if not files:
        parser.error("No .npz proposal files found")
    target.mkdir(parents=True)
    metadata = dict(input_dir=str(source), config=config.to_dict(),
                    score_note="Original SAM3 proposal scores retained; not recalibrated after cleanup.",
                    files=[])
    columns = ["image_id", "before_count", "after_count", "before_components", "after_components",
               "before_holes", "after_holes", "before_coverage", "after_coverage",
               "score_rejected", "area_rejected", "nms_rejected", "removed_island_pixels",
               "filled_hole_pixels", "status", "error"]
    errors = 0
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for number, path in enumerate(files, 1):
            row = dict(image_id=path.stem)
            try:
                with path.open("rb") as handle:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                    digest = hasher.hexdigest()
                    handle.seek(0)
                    with np.load(handle, allow_pickle=False) as data:
                        masks, scores = data["masks"], data["scores"]
                cleaned, new_scores, indices, changes = clean_proposals(masks, scores, config)
                before, after = proposal_stats(masks), proposal_stats(cleaned)
                save_atomic(target / path.name, masks=cleaned.astype(np.uint8),
                            scores=new_scores, source_indices=indices)
                row.update({f"before_{key}": value for key, value in before.items()})
                row.update({f"after_{key}": value for key, value in after.items()})
                row.update(changes, status="ok", error="")
                metadata["files"].append(dict(name=path.name, source_sha256=digest))
                print(f"[{number}/{len(files)}] {path.stem}: proposals {len(masks)} -> {len(cleaned)}, "
                      f"components {before['components']} -> {after['components']}", flush=True)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", 122)):
                    raise
                errors += 1
                row.update(status="error", error=str(exc))
                print(f"[ERROR] {path.stem}: {exc}", flush=True)
            writer.writerow(row)
            stream.flush()
    metadata.update(total=len(files), errors=errors)
    (target / "cleanup_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Done] {target}; {errors} errors. Coverage is not segmentation accuracy.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
