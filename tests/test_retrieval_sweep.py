import json
from pathlib import Path
import tempfile
import unittest

from tools import sweep_proposal_cleanup as sweep


class RetrievalSweepTests(unittest.TestCase):
    def result(self, case):
        return dict(mIoU=0.75, cleanup=case['cleanup'], evaluated_ids=['a'], evaluated_count=1,
                    feature_signature={}, feature_bank='bank', proposals_dir='proposals',
                    inference_settings={**case['retrieval'], 'INF_TOPK_NEIGH': 25},
                    proposal_count_before=100, proposal_count_after=10)

    def test_grid_is_unique_and_keeps_baseline(self):
        cases = sweep.retrieval_cases(0.8)
        self.assertEqual(len(cases), 29)
        self.assertEqual(cases[0]['name'], 'baseline_mlp3')
        self.assertEqual(len({json.dumps((c['cleanup'], c['retrieval']), sort_keys=True)
                              for c in cases}), 29)
        self.assertTrue(all(c['cleanup']['min_score'] == 0.8 for c in cases))

    def test_comparison_accepts_planned_changes_only(self):
        baseline, variant = sweep.retrieval_cases()[:2]
        reference, result = self.result(baseline), self.result(variant)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.json'
            sweep.write_json(path, result)
            sweep.read_result(path, variant, reference)
            result['inference_settings']['INF_TOPK_NEIGH'] = 99
            sweep.write_json(path, result)
            with self.assertRaisesRegex(ValueError, 'inference_settings'):
                sweep.read_result(path, variant, reference)
            result['inference_settings'] = reference['inference_settings']
            sweep.write_json(path, result)
            with self.assertRaisesRegex(ValueError, 'retrieval setting'):
                sweep.read_result(path, variant, reference)

    def test_ranking_exports_matched_cleanup_and_retrieval(self):
        cases = sweep.retrieval_cases()[:2]
        rows = [dict(self.result(c), name=c['name'], retrieval=c['retrieval'],
                     metrics_json='metrics.json') for c in cases]
        rows[1]['mIoU'] = 0.8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = sweep.save_ranking(root, rows, Path('proposals'))
            self.assertEqual(best['name'], cases[1]['name'])
            overrides = json.loads((root / 'best_retrieval_overrides.json').read_text())
            self.assertEqual(overrides['OVERRIDE_INF_CLASS_TOPM'], 1)
            self.assertIn('delta_percentage_points', (root / 'leaderboard.csv').read_text('utf-8-sig'))


if __name__ == '__main__':
    unittest.main()
