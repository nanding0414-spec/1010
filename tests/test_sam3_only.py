"""Regression checks for the SAM3-only deployment copy (no model download)."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from src import feature_extractor_factory as factory
from src.feature_bank_dataset import FeatureBankDataset, collate_fn
from tools import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class SAM3OnlyTests(unittest.TestCase):
    def config_process(self, **overrides):
        env = {key: value for key, value in os.environ.items() if not key.startswith("OVERRIDE_")}
        env.update(overrides)
        return subprocess.run(
            [sys.executable, "-B", "-c",
             "from configs.config import FEATURE_BACKBONE, OUTPUT_ROOT; "
             "print(FEATURE_BACKBONE); print(OUTPUT_ROOT)"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )

    def test_default_config_uses_sam3(self):
        result = self.config_process()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("\nsam3\n", result.stdout)
        self.assertIn("storage/raw_banks/voc/sam3_trunk_fpn_v1", result.stdout.replace('\\', '/'))

    def test_non_sam3_override_is_rejected(self):
        result = self.config_process(OVERRIDE_FEATURE_BACKBONE="unsupported-model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SAM3-only", result.stderr)

    def test_factory_builds_only_sam3(self):
        module = types.ModuleType("src.feature_extractor_sam3")
        module.SAM3FeatureExtractor = Mock()
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(factory, "FEATURE_ADAPTER_CKPT", ""):
            result = factory.build_feature_extractor(torch.device("cpu"))
        self.assertIs(result, module.SAM3FeatureExtractor.return_value)
        kwargs = module.SAM3FeatureExtractor.call_args.kwargs
        self.assertEqual(kwargs["feature_source"], factory.SAM3_FEATURE_SOURCE)
        self.assertFalse(kwargs["use_interactivity"])

    def test_factory_preserves_optional_adapter(self):
        module = types.ModuleType("src.feature_extractor_sam3")
        module.SAM3FeatureExtractor = Mock()
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch.object(factory, "FEATURE_ADAPTER_CKPT", "weights/test/adapter.pt"), \
                patch("src.feature_adapter.AdaptedFeatureExtractor") as adapter:
            result = factory.build_feature_extractor(torch.device("cpu"))
        adapter.assert_called_once_with(module.SAM3FeatureExtractor.return_value, "weights/test/adapter.pt")
        self.assertIs(result, adapter.return_value)

    def test_dataset_loads_saved_regions(self):
        regions = [{"cls_id": 1, "mask": "dummy"}]
        with patch("src.feature_bank_dataset.get_regions_from_pseudo", return_value=regions):
            dataset = FeatureBankDataset([Path("image.jpg")])
            self.assertEqual(dataset[0], ("image.jpg", regions))
            self.assertEqual(collate_fn([dataset[0], ("empty.jpg", [])]), [dataset[0]])

    def test_pipeline_cli_accepts_sam3_only(self):
        with patch.object(sys, "argv", ["run_pipeline.py", "voc", "--feature-backbone", "sam3"]):
            self.assertEqual(run_pipeline.parse_args().feature_backbone, "sam3")
        with patch.object(sys, "argv", ["run_pipeline.py", "voc", "--feature-backbone", "unsupported-model"]):
            with self.assertRaises(SystemExit) as error:
                run_pipeline.parse_args()
            self.assertEqual(error.exception.code, 2)

    def test_project_sources_parse(self):
        files = list(ROOT.glob("*.py"))
        for directory in ("src", "tools", "configs", "project_utils", "tests"):
            files.extend((ROOT / directory).rglob("*.py"))
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
