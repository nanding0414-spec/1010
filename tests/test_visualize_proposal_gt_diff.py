"""Synthetic diagnostics; no SAM3 accuracy claims."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from tools import visualize_proposal_gt_diff as diff


class CoverageTests(unittest.TestCase):
    def test_counts_colors_and_ignore(self):
        gt = np.array([[1, 0, 255], [2, 0, 255]], dtype=np.uint8)
        pred = np.array([[1, 1, 1], [0, 0, 0]], dtype=bool)
        regions, counts = diff.compare(pred, gt)
        self.assertEqual([counts[k] for k in ("tp", "fp", "fn")], [1, 1, 1])
        self.assertEqual(counts["iou"], 1 / 3)
        self.assertEqual(counts["precision"], .5)
        self.assertEqual(counts["ignored_pixels"], 2)
        blended = np.array(diff.overlay(np.zeros((2, 3, 3), dtype=np.uint8), regions))
        for key, point in {"tp": (0, 0), "fp": (0, 1), "fn": (1, 0), "ignore": (0, 2)}.items():
            np.testing.assert_array_equal(blended[point], np.rint(np.array(diff.COLORS[key]) * .5))

    def test_perfect_empty_and_undefined(self):
        gt = np.array([[1, 0]], dtype=np.uint8)
        self.assertEqual(diff.compare(gt == 1, gt)[1]["iou"], 1)
        counts = diff.compare(np.zeros_like(gt, dtype=bool), gt)[1]
        self.assertEqual(counts["fn"], 1)
        self.assertIsNone(counts["precision"])
        self.assertEqual(counts["recall"], 0)
        for gt in (np.zeros((1, 2), dtype=np.uint8), np.full((1, 2), 255, dtype=np.uint8)):
            self.assertIsNone(diff.compare(np.zeros((1, 2), dtype=bool), gt)[1]["iou"])

    def test_union_cache_validation_and_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.npz"
            masks = np.array([[[1, 0], [1, 0]], [[1, 1], [0, 0]]], dtype=np.uint8)
            np.savez(path, masks=masks, scores=[.1, .9])
            union, count = diff.load_union(path, (2, 2))
            self.assertEqual((count, int(union.sum())), (2, 3))
            with self.assertRaises(ValueError):
                diff.load_union(path, (3, 3))
            for bad_masks, scores in [(masks * 2, [.1, .9]), (masks, [np.nan, .1]), (masks, [.1])]:
                np.savez(path, masks=bad_masks, scores=scores)
                with self.assertRaises(ValueError):
                    diff.load_union(path, (2, 2))
            np.savez(path, masks=np.zeros((0, 1, 1)), scores=np.zeros(0))
            union, count = diff.load_union(path, (2, 3))
            self.assertEqual(union.shape, (2, 3))
            self.assertFalse(union.any())

    def test_invalid_gt(self):
        for gt in (np.array([[21]]), np.zeros((1, 1, 3)), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                diff.compare(np.zeros((1, 1), dtype=bool), gt)

    def test_extra_and_nested_proposals_are_visible(self):
        rgb = np.full((100, 480, 3), 200, dtype=np.uint8)
        gt = np.zeros((100, 480), dtype=np.uint8)
        gt[10:90, 10:90] = 1
        masks = np.zeros((3, 100, 480), dtype=bool)
        masks[0, 10:90, 10:90] = True
        masks[1, 10:90, 150:230] = True  # no GT overlap
        masks[2, 30:70, 170:210] = True  # nested; invisible in union alone
        regions, metrics = diff.compare(masks.any(axis=0), gt)
        self.assertEqual(metrics["fp"], 80 * 80)
        rendered = np.array(diff.render(rgb, masks, regions, "extra", metrics))
        self.assertEqual(rendered.shape, (226, 480, 3))
        np.testing.assert_array_equal(rendered[126 + 89, 180], [255, 220, 0])
        np.testing.assert_array_equal(rendered[126 + 69, 185], [255, 220, 0])
        np.testing.assert_array_equal(rendered[126 + 80, 180],
                                      np.rint(.5 * rgb[80, 180] + .5 * np.array(diff.COLORS["fp"])))

    def test_end_to_end_failures_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("images", "gt", "proposals"):
                (root / name).mkdir()
            for name in ("good", "bad", "missing", "dimensions", "labels"):
                Image.new("RGB", (3, 2), "white").save(root / "images" / f"{name}.png")
                gt = np.array([[1, 0, 255], [1, 0, 0]], dtype=np.uint8)
                if name == "labels":
                    gt[0, 0] = 21
                if name != "missing":
                    Image.fromarray(gt).save(root / "gt" / f"{name}.png")
                shape = (1, 2, 3) if name != "dimensions" else (1, 1, 1)
                np.savez(root / "proposals" / f"{name}.npz", masks=np.ones(shape), scores=[.1])
            (root / "proposals/bad.npz").write_bytes(b"invalid")
            args = ["--proposals-dir", str(root / "proposals"), "--image-dir", str(root / "images"),
                    "--gt-dir", str(root / "gt"), "--output-dir", str(root / "out")]
            self.assertEqual(diff.main(args), 1)
            run = json.loads((root / "out/run.json").read_text(encoding="utf-8"))
            self.assertEqual((run["succeeded"], run["failed"], run["status"]), (1, 4, "incomplete"))
            self.assertEqual((run["aggregate"]["tp"], run["aggregate"]["fp"]), (2, 3))
            with Image.open(root / "out/overlays/good.png") as im:
                self.assertEqual(im.size, (480, 128))
            self.assertFalse((root / "out/comparisons").exists())
            with self.assertRaises(FileExistsError):
                diff.main(args)
            args[-1] = str(root / "single")
            self.assertEqual(diff.main(args + ["--image-id", "good"]), 0)
            self.assertEqual(json.loads((root / "single/run.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
