import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.adapter_training import (image_id_from_sample, split_by_image,
                                  balanced_batch, retrieval_metrics)
from src.feature_signature import save_feature_signature, validate_feature_signature
from src.feature_bank_storage import save_feature_matrix_npz, load_feature_matrix_npz
from tests.test_feature_signature import _signature

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.feature_adapter import (FeatureAdapter, AdaptedFeatureExtractor,
                                     supervised_contrastive_loss, project_features,
                                     load_adapter, adapted_signature,
                                     CosineClassifier, feature_consistency_loss)


class AdapterDataTests(unittest.TestCase):
    def test_image_split_prevents_region_leakage(self):
        groups = np.repeat(["2007_000001", "2007_000002", "2007_000003", "2007_000004"], 2)
        labels = np.tile([0, 1], 4)
        train, val = split_by_image(groups, labels, 0.25, 42)
        self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertEqual(len(train) + len(val), len(groups))
        self.assertEqual(image_id_from_sample("2007_000001_000034"), "2007_000001")

    def test_positive_sampling_uses_distinct_images(self):
        groups = np.array(["a", "a", "b", "c", "d", "d", "e", "f"])
        labels = np.array([0] * 4 + [1] * 4)
        rows = balanced_batch(labels, groups, np.arange(8), np.random.default_rng(0), 2, 3)
        for cls in (0, 1):
            selected = rows[labels[rows] == cls]
            self.assertEqual(len(selected), 3)
            self.assertEqual(len(set(groups[selected])), 3)

    def test_retrieval_held_out_labels(self):
        features = np.array([[1., 0.], [0., 1.]])
        labels = np.array([0, 1])
        metrics = retrieval_metrics(features, labels, features[::-1], labels[::-1], k=1)
        self.assertEqual(metrics["macro_accuracy"], 1.0)

    def test_balanced_selection_uses_class_scores_not_majority_votes(self):
        gallery = np.vstack([[[1., 0.]], np.repeat([[.9, np.sqrt(1 - .9 ** 2)]], 20, axis=0)])
        labels = np.array([0] + [1] * 20)
        query = np.array([[1., 0.]])
        knn = retrieval_metrics(gallery, labels, query, np.array([0]), k=5)
        balanced = retrieval_metrics(gallery, labels, query, np.array([0]), k=25, mode="balanced")
        self.assertEqual(knn["accuracy"], 0.)
        self.assertEqual(balanced["accuracy"], 1.)
        self.assertEqual(balanced["per_class_accuracy"], {"0": 1.})

    def test_adapter_signatures_are_not_interchangeable(self):
        raw = _signature()
        first = {**raw, "adapter": {"sha256": "first"}}
        second = {**raw, "adapter": {"sha256": "second"}}
        with tempfile.TemporaryDirectory() as tmp:
            save_feature_signature(tmp, first)
            with self.assertRaisesRegex(RuntimeError, "adapter"):
                validate_feature_signature(tmp, second)
            with self.assertRaisesRegex(RuntimeError, "adapter"):
                validate_feature_signature(tmp, raw)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class AdapterModelTests(unittest.TestCase):
    def test_legacy_two_layer_checkpoint_preserves_bank_signature(self):
        model = FeatureAdapter(4, 8, 3, num_layers=2).eval()
        self.assertNotIn("num_layers", model.architecture)
        signature = {**_signature(), "embed_dim": 4}
        x = torch.randn(5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.pt"
            torch.save({"format_version": 1, "architecture": model.architecture,
                        "state_dict": model.state_dict(), "source_signature": signature}, path)
            original_signature = adapted_signature(signature, path, model.architecture)
            loaded, _ = load_adapter(path, "cpu", signature)
            torch.testing.assert_close(loaded(x), model(x))
            self.assertEqual(adapted_signature(signature, path, loaded.architecture),
                             original_signature)

    def test_cosine_classifier_and_perturbation_gradients(self):
        torch.manual_seed(3)
        model = FeatureAdapter(16, 8, 4)
        classifier = CosineClassifier(4, 3)
        x = torch.randn(12, 16)
        z = model(x)
        torch.testing.assert_close(classifier(z), classifier(3 * z))
        unchanged = feature_consistency_loss(model, x, z, 0.)
        self.assertAlmostEqual(unchanged.item(), 0., places=6)
        loss = feature_consistency_loss(model, x, z, .2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_training_cli_exports_usable_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            signature = {**_signature(), "embed_dim": 4}
            save_feature_signature(raw / "meta", signature)
            (raw / "meta" / "class_mapping.json").write_text(json.dumps({
                "id_to_dirname": {"0": "background", "1": "foreground"}
            }), encoding="utf-8")
            images = [f"image_{i:04d}" for i in range(12)]
            train_list = root / "train.txt"
            train_list.write_text("\n".join(images), encoding="utf-8")
            ids = [f"{name}_{i:06d}" for i, name in enumerate(images)]
            rng = np.random.default_rng(1)
            for cls, center in [("background", [1., 0., 0., 0.]),
                                ("foreground", [0., 1., 0., 0.])]:
                features = np.asarray(center) + rng.normal(0, .05, size=(12, 4))
                save_feature_matrix_npz(raw / "features" / cls / "features.npz", features, ids)
            script = Path(__file__).resolve().parents[1] / "tools" / "train_feature_adapter.py"
            for recipe in ("v1", "v2"):
                with self.subTest(recipe=recipe):
                    weights = root / f"weights_{recipe}"
                    bank = root / f"adapted_{recipe}"
                    result = subprocess.run([
                        sys.executable, str(script), "--source-bank", str(raw),
                        "--train-list", str(train_list), "--output-dir", str(weights),
                        "--export-bank", str(bank), "--epochs", "2", "--recipe", recipe,
                        "--steps-per-epoch", "2", "--hidden-dim", "8", "--output-dim", "3",
                        "--device", "cpu"
                    ], capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    projected, exported_ids = load_feature_matrix_npz(
                        bank / "features" / "foreground" / "features.npz")
                    self.assertEqual(projected.shape, (12, 3))
                    self.assertEqual(list(exported_ids), ids)
                    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1, atol=1e-5)
                    checkpoint = weights / "adapter.pt"
                    model, payload = load_adapter(checkpoint, "cpu", signature)
                    self.assertEqual(model.architecture.get("num_layers", 2), 2)
                    self.assertEqual(sum(isinstance(m, torch.nn.Linear)
                                         for m in model.net), 2)
                    self.assertEqual(payload["train_args"]["classifier"],
                                     "cosine" if recipe == "v2" else "linear")
                    validate_feature_signature(bank / "meta",
                                               adapted_signature(signature, checkpoint, model.architecture))

    def test_contrastive_loss_rewards_same_class_alignment(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        bad = good.detach()[[0, 2, 1, 3]].requires_grad_()
        good_loss = supervised_contrastive_loss(good, labels)
        self.assertLess(good_loss.item(), supervised_contrastive_loss(bad, labels).item())
        good_loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        empty_loss = supervised_contrastive_loss(bad, torch.arange(4))
        self.assertEqual(empty_loss.item(), 0.0)

    def test_empty_features_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        model = FeatureAdapter(4, 8, 3).eval()
        x = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [np.nan, 0, 0, 1]], dtype=np.float32)
        expected = project_features(model, x, "cpu")
        np.testing.assert_array_equal(expected[1:], 0)
        self.assertAlmostEqual(float(np.linalg.norm(expected[0])), 1.0, places=5)
        signature = {**_signature(), "embed_dim": 4}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.pt"
            torch.save({"format_version": 1, "architecture": model.architecture,
                        "state_dict": model.state_dict(), "source_signature": signature}, path)
            loaded, _ = load_adapter(path, "cpu", signature)
            np.testing.assert_allclose(project_features(loaded, x, "cpu"), expected)
            with self.assertRaisesRegex(ValueError, "signature"):
                load_adapter(path, "cpu", {**signature, "feature_source": "fpn"})

            class FakeBase:
                device = "cpu"
                def feature_signature(self):
                    return signature
                def extract_features(self, *_):
                    return torch.tensor(x[0])
                def extract_features_batch(self, *_):
                    return list(x)
                def extract_features_for_regions(self, *_):
                    return {1: list(x)}

            wrapper = AdaptedFeatureExtractor(FakeBase(), path)
            np.testing.assert_allclose(wrapper.extract_features(None, None).numpy(), expected[0])
            np.testing.assert_allclose(wrapper.extract_features_batch(None, None), expected)
            np.testing.assert_allclose(wrapper.extract_features_for_regions(None, None)[1], expected)
            self.assertEqual(wrapper.feature_signature(), adapted_signature(signature, path, model.architecture))


if __name__ == "__main__":
    unittest.main()
