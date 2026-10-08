"""CPU tests for paired seeds; no claimed model accuracy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import experimental_gmp_seeds as seeds


class SeedTests(unittest.TestCase):
    def test_only_seed_changes_and_split_is_required(self):
        args=['trainer.py','--seed','42','--split-file','fixed.json','--epochs','50']
        result=seeds.training_argv(args,123)
        self.assertEqual(result,['trainer.py','--seed','123','--split-file','fixed.json','--epochs','50'])
        self.assertEqual(args[2],'42')
        with self.assertRaises(ValueError):seeds.training_argv(['--seed','42'],123)
        with self.assertRaises(ValueError):seeds.training_argv(['--seed','7','--split-file','fixed.json'],123)

    def test_per_image_counts_and_ignore(self):
        pred=[np.array([[0,1],[1,0]]),np.array([[1,1]])]
        gt=[np.array([[0,0],[1,255]]),np.array([[1,0]])]
        result=seeds.image_confusions(pred,gt,2,255)
        np.testing.assert_array_equal(result,[[[1,1],[0,1]],[[0,1],[0,1]]])
        with self.assertRaises(ValueError):seeds.image_confusions([np.array([3])],[np.array([0])],2,255)

    def test_per_image_file_must_match_full_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'images.npz';matrices=np.zeros((2,21,21),dtype=np.int64)
            matrices[0,0,0]=5;matrices[1,1,1]=3
            np.savez_compressed(file,ids=['a','b'],confusions=matrices)
            metrics=dict(evaluated_ids=['a','b'],confusion_matrix=matrices.sum(0).tolist())
            seeds.validate_image_matrices(file,metrics)
            metrics['evaluated_ids']=['b','a']
            with self.assertRaises(ValueError):seeds.validate_image_matrices(file,metrics)

    def test_paired_statistics(self):
        reports={seed:dict(smfa_miou=.85,gmp_miou=.85+delta/100,delta_pp=delta,
                 per_class_iou={'chair':{'delta_pp':delta}}) for seed,delta in [(42,.2),(123,.1),(2026,.3)]}
        value=seeds.summary(reports)
        self.assertAlmostEqual(value['mean_delta_pp'],.2)
        self.assertAlmostEqual(value['sample_std_delta_pp'],.1)
        self.assertEqual(value['positive_seeds'],3)

    def test_worker_routes_training_to_requested_seed(self):
        for mode in ('smfa','gmp'):
            with tempfile.TemporaryDirectory() as directory:
                def fake_worker(job):
                    seeds.sys.argv=['trainer.py','--seed','42','--split-file','same.json']
                    seeds.ab.runpy.run_path(str(seeds.ROOT/'tools/train_feature_adapter.py'),run_name='__main__')
                with patch.object(seeds.ab.runpy,'run_path') as run,patch.object(seeds.sys,'argv',[]),\
                     patch.object(seeds.ab,'worker',side_effect=fake_worker),patch.object(seeds.gmp,'worker',side_effect=fake_worker):
                    seeds.worker(dict(stage='train',mode=mode,seed=2026,attempt=directory))
                    run.assert_called_once()
                    effective=seeds.ab.read(Path(directory)/'effective_training_argv.json')
                    self.assertEqual(effective['argv'][2],'2026')

    def test_worker_rejects_build(self):
        with self.assertRaises(ValueError):seeds.worker(dict(stage='build'))


if __name__=='__main__':unittest.main()
