import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.organize_storage import discover, migrate


class StorageTests(unittest.TestCase):
    def test_classification_uses_signature_and_ignores_canonical_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, signature in [('legacy/raw', {}), ('legacy/bank', {'adapter': {'type': 'mlp'}}),
                                    ('storage/raw_banks/voc/done', {})]:
                bank = root / name
                (bank / 'meta').mkdir(parents=True)
                (bank / 'features').mkdir()
                (bank / 'meta/feature_signature.json').write_text(json.dumps(signature))
            (root / 'data/VOC2012/pseudo/active').mkdir(parents=True)
            (root / 'proposals_old').mkdir()
            rows = discover(root)
            self.assertEqual(sorted(r['category'] for r in rows),
                             ['mlp_banks', 'proposals', 'pseudo_labels', 'raw_banks'])
            self.assertTrue(all(not Path(r['target']).exists() for r in rows))

    def test_collision_rejected_before_any_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            target.mkdir()
            with self.assertRaises(FileExistsError):
                migrate([dict(source=str(source), target=str(target))], root / 'journal')
            self.assertTrue(source.is_dir())
            self.assertFalse((root / 'journal').exists())

    def test_failed_symlink_rolls_back_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            (source / 'data').write_bytes(b'unchanged')
            with patch.object(Path, 'symlink_to', side_effect=OSError('denied')):
                with self.assertRaises(OSError):
                    migrate([dict(source=str(source), target=str(target))], root / 'journal')
            self.assertEqual((source / 'data').read_bytes(), b'unchanged')
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
