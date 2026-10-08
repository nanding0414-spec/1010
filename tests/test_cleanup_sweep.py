import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import run_inference as inference
from src.inference import predict_instances
from src.proposal_cleanup import CleanupConfig, clean_proposals
from tools import sweep_proposal_cleanup as sweep
from tools.refine_proposals import main as refine


def result_for(case, score=0.75):
    return {
        "mIoU": score, "cleanup": case["cleanup"], "evaluated_ids": ["sample"], "evaluated_count": 1,
        "feature_signature": {"backbone": "sam3"}, "inference_settings": {"MIN_MASK_AREA": 100},
        "feature_bank": "bank", "proposals_dir": "proposals",
        "proposal_count_before": 10, "proposal_count_after": 5,
    }


class CleanupSweepTests(unittest.TestCase):
    def test_defaults_include_raw_baseline_and_15_unique_cleanups(self):
        cases = sweep.load_cases()
        self.assertEqual(len(cases), 16)
        self.assertIsNone(cases[0]["cleanup"])
        self.assertEqual(len({json.dumps(case["cleanup"], sort_keys=True) for case in cases}), 16)

    def test_different_evaluation_images_cannot_be_ranked(self):
        case = sweep.default_cases()[0]
        reference = result_for(case)
        changed = dict(reference, evaluated_ids=["different"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            sweep.write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "evaluated_ids"):
                sweep.read_result(path, case, reference)

    def test_rank_uses_miou_not_mask_count_and_can_keep_raw(self):
        raw, cleaned = sweep.default_cases()[:2]
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(result_for(raw, 0.8), name=raw["name"], metrics_json="raw.json"),
                    dict(result_for(cleaned, 0.7), name=cleaned["name"], metrics_json="clean.json")]
            best = sweep.save_ranking(Path(directory), rows, Path("raw"))
            self.assertEqual(best["name"], "baseline_raw")
            self.assertIsNone(best["export_command"])
            self.assertIsNone(json.loads((Path(directory) / "best_cleanup_config.json").read_text()))
            rows[1]["mIoU"] = 0.9
            best = sweep.save_ranking(Path(directory), rows, Path("raw"))
            self.assertIn("--min-island-area 64", best["export_command"])

    def test_strict_proposal_validation_accepts_empty_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, masks=np.zeros((0, 1, 1), np.uint8), scores=np.array([]))
            self.assertEqual(inference.load_strict_proposals(directory, "sample", 5, 5), ([], []))
            np.savez(path, masks=np.zeros((1, 2, 2), np.uint8), scores=np.array([0.9]))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                inference.load_strict_proposals(directory, "sample", 5, 5)

    def test_strict_feature_errors_fail_instead_of_scoring_partial_output(self):
        extractor = Mock()
        extractor.extract_features_batch.side_effect = RuntimeError("test GPU failure")
        with self.assertRaisesRegex(RuntimeError, "test GPU failure"):
            predict_instances(np.zeros((4, 4, 3), np.uint8), extractor,
                              all_feats=np.ones((1, 2)), all_labels=["cat"],
                              precomputed_masks=[np.ones((4, 4), bool)], precomputed_scores=[0.9],
                              strict_errors=True)

    def test_in_memory_cleanup_equals_saved_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            masks = np.zeros((2, 8, 8), np.uint8)
            masks[0, 1:7, 1:7] = 1
            masks[1, 0, 0] = 1
            scores = np.array([0.9, 0.1], np.float32)
            np.savez(raw / "sample.npz", masks=masks, scores=scores)
            config = CleanupConfig(min_mask_area=2, min_island_area=2)
            command = ["--input-dir", str(raw), "--output-dir", str(root / "clean")]
            for key, value in config.to_dict().items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            self.assertEqual(refine(command), 0)
            in_memory, memory_scores, _, _ = clean_proposals(masks, scores, config)
            with np.load(root / "clean/sample.npz") as saved:
                np.testing.assert_array_equal(in_memory, saved["masks"])
                np.testing.assert_array_equal(memory_scores, saved["scores"])

    def test_inference_cli_applies_cleanup_and_writes_complete_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "sample.jpg")
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(root / "sample.png")
            masks = np.ones((2, 8, 8), np.uint8)
            np.savez(raw / "sample.npz", masks=masks, scores=np.array([0.9, 0.1]))
            config = CleanupConfig(min_mask_area=1, min_island_area=1)
            sweep.write_json(root / "cleanup.json", config.to_dict())
            extractor = Mock(embed_dim=2)
            extractor.feature_signature.return_value = {"backbone": "sam3", "embed_dim": 2}
            predictions = Mock(return_value=[])
            patches = {
                "INF_LOAD_PROPOSALS": True, "INF_PROPOSALS_DIR": str(raw),
                "INF_FEATURE_BANK_ROOT": str(root / "bank"), "INF_VIS_LIMIT": 0,
                "load_eval_samples": Mock(return_value=[{"id": "sample", "img": str(root / "sample.jpg"),
                                                        "gt": str(root / "sample.png")}]),
                "load_class_mapping": Mock(return_value=({}, ["cat"], [])),
                "load_class_features": Mock(return_value={"cat": np.array([[1.0, 0.0]])}),
                "build_feature_extractor": Mock(return_value=extractor),
                "validate_feature_signature": Mock(), "build_retrieval_context": Mock(),
                "build_faiss_index": Mock(return_value=None), "predict_instances": predictions,
                "rasterize_instances": Mock(return_value=np.zeros((8, 8), np.uint8)),
                "evaluate_dataset_miou": Mock(return_value=("test", {"mIoU": 1.0,
                    "per_class_iou": np.array([1.0, 1.0])}, ["background", "cat"])),
            }
            with ExitStack() as stack:
                for name, value in patches.items():
                    stack.enter_context(patch.object(inference, name, value))
                inference.main(["--strict-eval", "--cleanup-config", str(root / "cleanup.json"),
                                "--output-dir", str(root / "trial"),
                                "--results-json", str(root / "metrics.json")])
            self.assertEqual(len(predictions.call_args.kwargs["precomputed_masks"]), 1)
            self.assertTrue(predictions.call_args.kwargs["strict_errors"])
            result = json.loads((root / "metrics.json").read_text())
            self.assertEqual(result["evaluated_ids"], ["sample"])
            self.assertEqual(result["proposal_count_before"], 2)
            self.assertEqual(result["proposal_count_after"], 1)
            self.assertFalse((root / "bank").exists())

    def test_sweep_runs_and_resumes_without_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, bank, weights = root / "raw", root / "bank", root / "weights"
            raw.mkdir()
            (bank / "features/cat").mkdir(parents=True)
            (bank / "meta").mkdir()
            weights.mkdir()
            np.savez(raw / "sample.npz", masks=np.ones((1, 2, 2)), scores=np.array([0.9]))
            np.savez(bank / "features/cat/cat_features.npz", features=np.ones((1, 2)))
            adapter = weights / "adapter.pt"
            adapter.write_bytes(b"dummy adapter")
            checkpoint = weights / "sam3.pt"
            checkpoint.write_bytes(b"dummy sam3")
            signature = dict(backbone="sam3", feature_source="trunk_fpn", feature_level=-1,
                             resolution=1008, token_norm="layernorm", context_subtract=0,
                             use_bf16=False, adapter={"sha256": sweep.sha256(adapter)})
            sweep.write_json(bank / "meta/feature_signature.json", signature)
            sweep.write_json(bank / "meta/class_mapping.json", {})
            cases = sweep.default_cases()[:2]
            sweep.write_json(root / "cases.json", cases)
            args = ["--proposals-dir", str(raw), "--bank", str(bank), "--adapter", str(adapter),
                    "--output-dir", str(root / "output"), "--cases", str(root / "cases.json")]

            def fake_inference(command, **kwargs):
                case = cases[1] if "--cleanup-config" in command else cases[0]
                payload = result_for(case, 0.8 if case["cleanup"] else 0.7)
                payload["feature_signature"] = signature
                path = Path(command[command.index("--results-json") + 1])
                sweep.write_json(path, payload)
                return subprocess.CompletedProcess(command, 0)

            with patch.dict(os.environ, {"OVERRIDE_SAM3_CKPT": str(checkpoint)}), \
                    patch.object(sweep.subprocess, "run", side_effect=fake_inference) as run:
                sweep.main(args)
                self.assertEqual(run.call_count, 2)
                sweep.main(args + ["--resume"])
                self.assertEqual(run.call_count, 2)
                self.assertTrue((root / "output/sweep_complete.json").is_file())
                best = json.loads((root / "output/best.json").read_text())
                self.assertEqual(best["name"], "cleanup_default")
                self.assertEqual(len(list(raw.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
