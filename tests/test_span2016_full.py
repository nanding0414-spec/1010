import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.run_span2016_full import check_coverage, merge_metrics, parallel, merge_banks, write, read


class Full2016Tests(unittest.TestCase):
    def test_confusions_are_summed_not_averaged_and_ids_checked(self):
        matrices=[np.zeros((21,21),dtype=int) for _ in range(2)]
        matrices[0][0,0]=90; matrices[1][0,1]=10
        common=dict(feature_signature={},feature_bank='bank',cleanup={'min_score':.7},proposals_dir='p',
            inference_settings={},feature_image_dir='orig',feature_image_scale=1,
            per_class_iou={str(i):0 for i in range(21)},proposal_count_before=5,proposal_count_after=3)
        results=[dict(common,evaluated_ids=[str(i)],confusion_matrix=m.tolist()) for i,m in enumerate(matrices)]
        result=merge_metrics(results,['0','1'])
        self.assertAlmostEqual(result['per_class_iou']['0'],.9)
        self.assertAlmostEqual(result['mIoU'],.9/21)
        self.assertEqual(result['proposal_count_after'],6)
        with self.assertRaises(ValueError):
            merge_metrics(results,['0','1','2'])
        with self.assertRaises(ValueError):
            check_coverage([['a'],['a']],['a','b'])

    def test_parallel_errors_are_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=[([sys.executable,'-c','raise SystemExit(2)'],dict(os.environ),root/'fail.log'),
                  ([sys.executable,'-c','import time; time.sleep(30)'],dict(os.environ),root/'other.log')]
            with self.assertRaises(RuntimeError):
                parallel(jobs)
            self.assertTrue((root/'fail.command.json').is_file())

    def test_merge_unfiltered_banks_and_filter_once(self):
        from src.feature_bank_storage import save_feature_matrix_npz, load_feature_matrix_npz
        from configs import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); shards=[root/'s0',root/'s1']; target=root/'merged'
            for i,s in enumerate(shards):
                write(s/'meta/coverage.json',dict(ids=[str(i)]))
                write(s/'meta/feature_signature.json',{'resolution':2016})
                write(s/'meta/class_mapping.json',{'id_to_dirname':{'0':'background'}})
                save_feature_matrix_npz(s/'features/background/features.npz',np.ones((1,4)),[f'{i}_000000'])
            with patch.object(cfg,'OUTPUT_ROOT',str(target)), patch('src.filter_stage.run_filter_stage') as filtering, \
                 patch('project_utils.config_logger.record_config'):
                merge_banks(dict(shards=list(map(str,shards)),ids=['0','1']))
                filtering.assert_called_once()
            x,ids=load_feature_matrix_npz(target/'features/background/background_features.npz')
            self.assertEqual(x.shape,(2,4)); self.assertEqual(len(set(ids)),2)
            self.assertEqual(read(target/'meta/coverage.json')['ids'],['0','1'])


if __name__=='__main__':
    unittest.main()
