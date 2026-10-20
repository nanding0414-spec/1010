"""Publication contracts: fixed inputs and unchanged balanced retrieval results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import run_r3_record as cli
from src.retrieval import build_retrieval_context, classify_queries


class R3ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = cli.read_json(cli.PROFILE)
        self.bank = self.root / "bank"
        (self.bank / "features").mkdir(parents=True)
        (self.bank / "meta").mkdir()
        self.weights = self.root / "weights/sam3.pt"
        self.weights.parent.mkdir()
        self.weights.write_bytes(b"test-checkpoint")
        self.adapter = self.root / "adapter/model.pt"
        self.adapter.parent.mkdir()
        self.adapter.write_bytes(b"test-adapter")
        self.profile["source_signature"]["checkpoint_sha256"] = cli.worker.sha256(self.weights)
        signature = copy.deepcopy(self.profile["source_signature"])
        signature["embed_dim"] = 256
        signature["adapter"] = dict(sha256=cli.worker.sha256(self.adapter),
                                    architecture=dict(input_dim=1280, hidden_dim=512, output_dim=256))
        meta = self.bank / "meta/feature_signature.json"
        meta.write_text(json.dumps(signature), encoding="utf-8")
        self.manifest = dict(version="new_attempt", source_signature=self.profile["source_signature"],
                             recipe_id="r3_record_v1", seed=42, baseline_recipe=self.profile["baseline"],
                             sam3_weights=str(self.weights), source_adapter=str(self.adapter),
                             source_mlp_bank=str(self.bank), code_digest="frozen-code", parameter_digest="frozen-parameters",
                             input_artifacts=[cli.worker._identity(p) for p in [self.weights, self.adapter, meta]],
                             token_records={"id": dict(path=str(self.root/"tokens/id.npz"), sha256="on-consumption",
                                                      receipt_path=str(self.root/"tokens/id.json"), receipt_sha256="on-consumption",
                                                      geometry=dict(original_hw=[100, 80], token_hw=[72, 72]))},
                             image_records={"id": dict(path=str(self.root/"images/id.jpg"), sha256="on-consumption")},
                             baseline_records={"id": dict(path=str(self.root/"baseline/id.png"), sha256="on-consumption")})
        self.job = dict(arm="R3", stage="generate", image_ids=["id"], output_dir=str(self.root/"new/proposals"),
                        context_path=str(self.root/"new/context.pt"))

    def test_valid_preflight_does_not_load_tokens_models_or_GT(self):
        # Test data deliberately have no real tokens, images, torch weights or GT.
        cli.validate(self.profile, self.manifest, self.job)

    def test_different_arm_or_center_recipe_is_rejected(self):
        for change in [dict(arm="R2"), dict(stage="train")]:
            with self.assertRaises(ValueError):
                cli.validate(self.profile, self.manifest, dict(self.job, **change))
        self.manifest["baseline_recipe"] = dict(self.profile["baseline"], centroid_weight=.4)
        with self.assertRaisesRegex(ValueError, "baseline recipe"):
            cli.validate(self.profile, self.manifest, self.job)

    def test_tampered_input_rejected_before_worker(self):
        self.adapter.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA/size"):
            cli.validate(self.profile, self.manifest, self.job)

    def test_bank_adapter_binding_mismatch_is_rejected(self):
        meta = self.bank / "meta/feature_signature.json"
        signature = cli.read_json(meta)
        signature["adapter"]["sha256"] = "wrong-model"
        meta.write_text(json.dumps(signature), encoding="utf-8")
        self.manifest["input_artifacts"][-1] = cli.worker._identity(meta)
        with self.assertRaisesRegex(ValueError, "checkpoint binding"):
            cli.validate(self.profile, self.manifest, self.job)

    def test_output_cannot_overwrite_source_or_ancestor(self):
        for path in [self.root, self.bank, self.bank/"new", self.root/"baseline"]:
            with self.assertRaisesRegex(ValueError, "Output"):
                cli.validate(self.profile, self.manifest, dict(self.job, output_dir=str(path)))

    def test_missing_signature_and_duplicate_image_ids_rejected(self):
        for key in ["code_digest", "parameter_digest"]:
            manifest = dict(self.manifest)
            del manifest[key]
            with self.assertRaisesRegex(ValueError, "Missing signed manifest"):
                cli.validate(self.profile, manifest, self.job)
        with self.assertRaisesRegex(ValueError, "unique image"):
            cli.validate(self.profile, self.manifest, dict(self.job, image_ids=["id", "id"]))

    def test_upstream_from_other_attempt_is_not_reused(self):
        folder = self.root / "proposals/id"
        folder.mkdir(parents=True)
        (folder / "complete.json").write_text(json.dumps(dict(binding=dict(arm="R3", version="old"), artifacts=[])), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "version binding"):
            cli.validate(self.profile, self.manifest, dict(self.job, stage="query", candidates_dir=str(folder.parent)))

    def test_balanced_score_metadata_keeps_original_label_score_margin(self):
        gallery = np.asarray([[1., 0.], [.8, .6], [0., 1.], [.6, .8]], np.float32)
        labels = ["chair", "chair", "sofa", "sofa"]
        context = build_retrieval_context(gallery, labels)
        query = np.asarray([[1., 0.]], np.float32)
        pred = classify_queries(query, gallery, labels, None, context, 25, "balanced", 5, 2, .2)[0]
        centers = (query @ context.centroids.T)[0]
        expected = np.asarray([.8 * float(gallery[:2, 0].mean()) + .2 * float(centers[0]),
                               .8 * float(gallery[2:, 0].mean()) + .2 * float(centers[1])], np.float32)
        np.testing.assert_array_equal(pred.class_scores, expected)
        self.assertEqual(pred.label, "chair")
        self.assertEqual(pred.score, float(expected[0]))
        self.assertEqual(pred.margin, float(expected[0]) - float(expected[1]))
        legacy = classify_queries(query, gallery, labels, None, context, 3, "legacy")[0]
        self.assertIsNone(legacy.class_scores)


if __name__ == "__main__":
    unittest.main()
