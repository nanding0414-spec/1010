import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from src.bank_input_validation import image_index, native_image_dir, validate_pairs
from project_utils.experiment import relocated


class BankInputTests(unittest.TestCase):
    def test_native_pairs_require_complete_matching_class_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root / 'images', root / 'masks'
            images.mkdir(); masks.mkdir()
            Image.new('RGB', (10, 8)).save(images / 'a.jpg')
            Image.new('L', (10, 8), 20).save(masks / 'a.png')
            self.assertEqual(validate_pairs(masks, images, ['a']), [images / 'a.jpg'])
            with self.assertRaises(FileNotFoundError):
                validate_pairs(masks, images, ['a', 'b'])
            Image.new('L', (5, 4)).save(masks / 'a.png')
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                validate_pairs(masks, images, ['a'])
            Image.new('L', (10, 8), 21).save(masks / 'a.png')
            with self.assertRaisesRegex(ValueError, 'class-ID'):
                validate_pairs(masks, images, ['a'])
            Image.new('RGB', (10, 8)).save(images / 'a.png')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                image_index(images)

    def test_metadata_and_archived_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / 'pseudo/meta/shard_0'
            meta.mkdir(parents=True)
            config = meta / 'generation.json'
            config.write_text(json.dumps({'image_dir': str(root / 'span')}))
            self.assertEqual(native_image_dir(root / 'pseudo'), (root / 'span').resolve())
            config.write_text(json.dumps({'image_dir': str(root / 'span'), 'output_size_reference_dir': 'original'}))
            with self.assertRaisesRegex(ValueError, 'resized'):
                native_image_dir(root / 'pseudo')
            target = root / 'experiments/old/metrics.json'
            target.parent.mkdir(parents=True)
            target.write_text('{}')
            self.assertEqual(relocated(root / 'old/metrics.json', root), target.resolve())
            journal = root / 'storage/migrations/move.jsonl'
            journal.parent.mkdir(parents=True)
            journal.write_text(json.dumps({'event': 'complete', 'source': str(root / 'old_bank'), 'target': str(target.parent)}))
            self.assertEqual(relocated(root / 'old_bank/metrics.json', root), target.resolve())


if __name__ == '__main__':
    unittest.main()
