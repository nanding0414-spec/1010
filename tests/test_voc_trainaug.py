import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.generate_voc_imagelevel_train_val import main


class VOCTrainaugTests(unittest.TestCase):
    def make_dataset(self, directory):
        root = Path(directory)
        splits = root / "ImageSets/Segmentation"
        splits.mkdir(parents=True)
        (splits / "train.txt").write_text("image_a\n", encoding="utf-8")
        (splits / "trainaug.txt").write_text("image_a\nimage_b\n", encoding="utf-8")
        (splits / "val.txt").write_text("image_c\n", encoding="utf-8")
        masks = root / "SegmentationClassAug"
        masks.mkdir()
        Image.fromarray(np.array([[0, 1], [4, 255]], np.uint8)).save(masks / "image_a.png")
        Image.fromarray(np.array([[0, 8], [12, 255]], np.uint8)).save(masks / "image_b.png")
        return root

    def test_trainaug_json_has_augmented_ids_and_only_class_presence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_dataset(directory)
            main(["--voc-root", str(root), "--splits", "trainaug"])
            payload = json.loads((root / "ImageSets/ImageLevel/trainaug_imagelevel.json").read_text())
            self.assertEqual([row["img_id"] for row in payload["images"]], ["image_a", "image_b"])
            self.assertEqual(payload["images"][0]["labels"], [0, 1, 4])
            self.assertEqual(payload["images"][1]["labels"], [0, 8, 12])
            self.assertFalse((root / "ImageSets/ImageLevel/train_imagelevel.json").exists())

    def test_missing_augmented_annotation_is_not_silent_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_dataset(directory)
            (root / "SegmentationClassAug/image_b.png").unlink()
            with self.assertRaises(FileNotFoundError):
                main(["--voc-root", str(root), "--splits", "trainaug"])
            self.assertFalse((root / "ImageSets/ImageLevel/trainaug_imagelevel.json").exists())

    def test_validation_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_dataset(directory)
            (root / "ImageSets/Segmentation/val.txt").write_text("image_b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation images"):
                main(["--voc-root", str(root), "--splits", "trainaug"])

    def test_config_explicit_split_and_trainaug_default(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, OVERRIDE_DATASET_TYPE="voc", OVERRIDE_FEATURE_ADAPTER_CKPT="")
        env.pop("OVERRIDE_VOC_TRAIN_SPLIT", None)
        command = [sys.executable, "-c", "from configs.config import DATASET_TRAIN_SPLIT, IMAGELEVEL_JSON_PATH; "
                   "print(DATASET_TRAIN_SPLIT); print(IMAGELEVEL_JSON_PATH)"]
        baseline = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
        self.assertIn("\ntrainaug\n", baseline.stdout)
        env["OVERRIDE_VOC_TRAIN_SPLIT"] = "trainaug"
        augmented = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
        self.assertIn("\ntrainaug\n", augmented.stdout)
        self.assertIn("trainaug_imagelevel.json", augmented.stdout)
        env["OVERRIDE_VOC_TRAIN_SPLIT"] = "val"
        invalid = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
