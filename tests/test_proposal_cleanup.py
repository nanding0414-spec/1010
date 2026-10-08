from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.proposal_cleanup import CleanupConfig, clean_mask, clean_proposals, mask_nms
from tools.refine_proposals import main, save_atomic


class ProposalCleanupTests(unittest.TestCase):
    def test_removes_islands_without_eroding_object(self):
        mask = np.zeros((100, 100), bool)
        mask[20:80, 20:80] = True
        mask[1:3, 1:3] = True
        cleaned, removed, _ = clean_mask(mask, CleanupConfig())
        self.assertEqual(removed, 4)
        self.assertTrue(cleaned[20:80, 20:80].all())
        self.assertFalse(cleaned[1:3, 1:3].any())

    def test_fills_only_tiny_enclosed_holes(self):
        mask = np.ones((100, 100), bool)
        mask[10:12, 10:12] = False
        mask[30:50, 30:50] = False
        mask[0:2, 70:72] = False
        result, _, filled = clean_mask(mask, CleanupConfig())
        self.assertEqual(filled, 4)
        self.assertTrue(result[10:12, 10:12].all())
        self.assertFalse(result[30:50, 30:50].any())
        self.assertFalse(result[0:2, 70:72].any())

    def test_hole_fraction_protects_small_ring(self):
        mask = np.zeros((20, 20), bool)
        mask[2:18, 2:18] = True
        mask[8:12, 8:12] = False
        cleaned, _, filled = clean_mask(mask, CleanupConfig())
        self.assertEqual(filled, 0)
        np.testing.assert_array_equal(cleaned, mask)

    def test_nms_keeps_nested_small_object_and_distinct_shapes(self):
        large = np.ones((20, 20), bool)
        small = np.zeros_like(large)
        small[3:7, 3:7] = True
        self.assertEqual(mask_nms([large, large.copy(), small], [.8, .9, .7], .85), [1, 2])
        diagonal = np.eye(20, dtype=bool)
        self.assertEqual(mask_nms([diagonal, np.fliplr(diagonal)], [.9, .8], .5), [0, 1])

    def test_preserves_scores_ids_and_small_high_score_object(self):
        masks = np.zeros((3, 30, 30), np.uint8)
        masks[:, 5:15, 5:15] = 1
        scores = np.array([.05, .9, .8], np.float32)
        cleaned, new_scores, indices, report = clean_proposals(masks, scores, CleanupConfig())
        self.assertEqual(len(cleaned), 1)
        np.testing.assert_array_equal(indices, [1])
        np.testing.assert_array_equal(new_scores, scores[indices])
        self.assertEqual(report["score_rejected"], 1)
        self.assertEqual(report["nms_rejected"], 1)
        self.assertEqual(cleaned.sum(), 100)
        self.assertEqual(masks.sum(), 300)

    def test_empty_invalid_and_all_fragmented(self):
        empty, scores, ids, _ = clean_proposals(np.zeros((0, 1, 1)), [], CleanupConfig())
        self.assertEqual(empty.shape, (0, 1, 1))
        self.assertEqual(len(scores), len(ids))
        fragmented = np.zeros((1, 100, 100), np.uint8)
        fragmented[0, ::3, ::3] = 1
        cleaned, _, _, report = clean_proposals(fragmented, [.9], CleanupConfig())
        self.assertEqual(cleaned.shape, (0, 100, 100))
        self.assertEqual(report["area_rejected"], 1)
        with self.assertRaises(ValueError):
            clean_proposals(fragmented, [float("nan")], CleanupConfig())

    def test_atomic_failure_does_not_leave_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            with mock.patch("tools.refine_proposals.np.savez_compressed", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_atomic(path, scores=np.ones(1))
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_cli_is_compatible_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw", Path(directory) / "new"
            source.mkdir()
            path = source / "sample.npz"
            np.savez_compressed(path, masks=np.ones((1, 20, 20), np.uint8), scores=np.array([.9]))
            original = path.read_bytes()
            args = ["--input-dir", str(source), "--output-dir", str(output)]
            self.assertEqual(main(args), 0)
            with np.load(output / path.name) as data:
                self.assertEqual(data["masks"].shape, (1, 20, 20))
                np.testing.assert_array_equal(data["source_indices"], [0])
            self.assertTrue((output / "summary.csv").is_file())
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
