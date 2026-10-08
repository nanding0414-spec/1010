import tempfile
import unittest
from pathlib import Path

from src.feature_signature import save_feature_signature, validate_feature_signature


def _signature(source="trunk_fpn"):
    return {
        "format_version": 2,
        "backbone": "sam3",
        "feature_source": source,
        "feature_level": -1,
        "resolution": 1008,
        "token_norm": "layernorm",
        "context_subtract": 0.0,
        "embed_dim": 1280,
        "feature_grid": [72, 72],
        "use_bf16": True,
        "model_name": "sam3",
    }


class FeatureSignatureTests(unittest.TestCase):
    def test_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_feature_signature(tmp, _signature())
            validate_feature_signature(tmp, _signature())

    def test_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_feature_signature(tmp, _signature("fpn"))
            with self.assertRaisesRegex(RuntimeError, "different feature spaces"):
                validate_feature_signature(tmp, _signature("trunk_fpn"))

    def test_missing_signature_requires_explicit_legacy_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(Path(tmp).is_dir())
            with self.assertRaisesRegex(RuntimeError, "Rebuild"):
                validate_feature_signature(tmp, _signature())
            validate_feature_signature(tmp, _signature(), allow_missing=True)


if __name__ == "__main__":
    unittest.main()
