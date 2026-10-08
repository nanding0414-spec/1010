import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.feature_bank_storage import save_feature_matrix_npz
from tools import visualize_feature_bank as visual


class FeatureBankVisualizationTests(unittest.TestCase):
    def make_bank(self, root, dim, shift=0):
        (root / 'meta').mkdir(parents=True)
        (root / 'meta/feature_signature.json').write_text(json.dumps({'embed_dim': dim}))
        rng = np.random.default_rng(1)
        for i, name in enumerate(['cat', 'dog', 'horse']):
            x = rng.normal(0, 0.15, (12, dim)).astype(np.float32)
            x[:, i] += 1
            ids = [f'image_{j:04d}_{i:06d}' for j in range(shift, shift+12)]
            save_feature_matrix_npz(root / 'features' / name / 'features.npz', x, ids)

    def test_common_sampling_matches_ids_across_different_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bank(root / 'raw', 8)
            self.make_bank(root / 'adapted', 4, shift=3)
            banks = [visual.scan_bank('raw', root / 'raw'), visual.scan_bank('adapted', root / 'adapted')]
            chosen, excluded = visual.select_samples(banks, 5, 42)
            self.assertFalse(excluded)
            first, y1, ids1 = visual.load_selected(banks[0], chosen, 0)
            second, y2, ids2 = visual.load_selected(banks[1], chosen, 1)
            self.assertEqual(ids1, ids2)
            np.testing.assert_array_equal(y1, y2)
            self.assertEqual(first.shape, (15, 8))
            self.assertEqual(second.shape, (15, 4))
            np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)

    def test_invalid_vectors_removed_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'features.npz'
            x = np.array([[1., 0.], [0., 0.], [np.nan, 1.], [np.inf, 1.]])
            save_feature_matrix_npz(path, x, ['a', 'b', 'c', 'd'])
            valid, ids, invalid = visual.load_class(path)
            self.assertEqual(invalid, 3)
            self.assertEqual(ids.tolist(), ['a'])
            save_feature_matrix_npz(path, valid.repeat(2, axis=0), ['a', 'a'])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                visual.load_class(path)

    def test_centroid_stats_known_orthogonal_classes(self):
        x = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        similarity, stats = visual.cohort_statistics(x, np.array([0, 0, 1, 1]), ['cat', 'dog'])
        np.testing.assert_allclose(similarity, np.eye(2))
        for row in stats:
            self.assertAlmostEqual(row['cohesion'], 1.)
            self.assertAlmostEqual(row['mean_centroid_margin'], 1.)
            self.assertEqual(row['negative_margin_fraction'], 0.)

    def test_cli_renders_all_artifacts_and_preserves_inputs(self):
        try:
            import matplotlib
            import sklearn
        except ImportError:
            self.skipTest('requires matplotlib and scikit-learn')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bank(root / 'raw', 8)
            self.make_bank(root / 'adapted', 4, shift=3)
            files = list((root / 'raw').rglob('*.npz'))
            before = [p.read_bytes() for p in files]
            out = root / 'out'
            args = ['--bank', f'raw={root / "raw"}', '--bank', f'adapted={root / "adapted"}',
                    '--output-dir', str(out), '--max-per-class', '5', '--iterations', '300', '--threads', '1']
            visual.main(args)
            for name in ('pca_comparison.png', 'tsne_comparison.png', 'centroid_similarity.png',
                         'class_diagnostics.png', 'class_statistics.csv', 'summary.json'):
                self.assertGreater((out / name).stat().st_size, 100)
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['banks'][0]['sampled_count'], 15)
            self.assertEqual(len(report['classes']), 3)
            self.assertEqual(before, [p.read_bytes() for p in files])
            with self.assertRaises(SystemExit):
                visual.main(args)


if __name__ == '__main__':
    unittest.main()
