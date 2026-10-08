import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.visualize_proposals import (
    load_cached_proposals, main, proposal_color, render_overlay, select_proposals,
)


class ProposalVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((80, 100, 3), 120, dtype=np.uint8)
        self.masks = np.zeros((3, 80, 100), dtype=bool)
        self.masks[0, 10:60, 10:70] = True
        self.masks[1, 30:70, 40:90] = True
        self.scores = np.array([0.4, 0.9, 0.8], dtype=np.float32)

    def test_filters_are_optional_and_do_not_change_cache(self):
        np.testing.assert_array_equal(select_proposals(self.masks, self.scores), [1, 0])
        np.testing.assert_array_equal(select_proposals(self.masks, self.scores, 0.5), [1])
        np.testing.assert_array_equal(select_proposals(self.masks, self.scores, top_k=1), [1])
        self.assertEqual(int(self.masks[0].sum()), 3000)

    def test_overlay_blends_once_and_preserves_background(self):
        indices = select_proposals(self.masks, self.scores)
        result, coverage = render_overlay(self.rgb, self.masks, self.scores, indices, alpha=0.4)
        np.testing.assert_array_equal(result[0, 0], self.rgb[0, 0])
        expected = np.rint(self.rgb[45, 50] * 0.6 + np.array(proposal_color(1)) * 0.4)
        np.testing.assert_array_equal(result[45, 50], expected)
        np.testing.assert_array_equal(coverage, self.masks.any(axis=0))
        np.testing.assert_array_equal(self.rgb, np.full_like(self.rgb, 120))
        self.assertFalse(np.array_equal(result[10, 20], self.rgb[10, 20]))

    def test_empty_legacy_cache_and_size_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.npz"
            np.savez_compressed(path, masks=np.zeros((0, 1, 1), np.uint8), scores=np.zeros(0))
            masks, scores = load_cached_proposals(path, (80, 100))
            indices = select_proposals(masks, scores)
            result, coverage = render_overlay(self.rgb, masks, scores, indices)
            np.testing.assert_array_equal(result, self.rgb)
            self.assertFalse(coverage.any())
            np.savez_compressed(path, masks=self.masks, scores=self.scores)
            with self.assertRaisesRegex(ValueError, "Mask size"):
                load_cached_proposals(path, (100, 80))

    def test_cli_outputs_and_reports_missing_or_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, proposals, output = [root / name for name in ("images", "proposals", "out")]
            images.mkdir()
            proposals.mkdir()
            for name in ("sample", "missing", "corrupt"):
                Image.fromarray(self.rgb).save(images / f"{name}.jpg")
            np.savez_compressed(proposals / "sample.npz", masks=self.masks, scores=self.scores)
            (proposals / "corrupt.npz").write_bytes(b"PK\x03\x04broken")
            args = ["--image-dir", str(images), "--proposals-dir", str(proposals),
                    "--output-dir", str(output)]
            self.assertEqual(main(args + ["--image-id", "sample", "--save-individual",
                                          "--draw-labels"]), 0)
            with Image.open(output / "overlays/sample.jpg") as image:
                self.assertEqual(image.size, (100, 80))
            with Image.open(output / "comparisons/sample.jpg") as image:
                self.assertEqual(image.size, (200, 80))
            self.assertEqual(len(list((output / "individual/sample").glob("*.jpg"))), 2)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
                self.assertEqual(row["cached_count"], "3")
                self.assertEqual(row["shown_count"], "2")
            for name in ("missing", "corrupt"):
                self.assertEqual(main(args + ["--image-id", name]), 1)


if __name__ == "__main__":
    unittest.main()
