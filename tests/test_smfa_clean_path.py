"""Regression checks for the retained region path and removed feature spaces."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np


class CleanPathTests(unittest.TestCase):
    def test_missing_spatial_tokens_never_falls_back_to_global_average(self):
        from tests.test_feature_pooling import DummySpatialExtractor
        model=DummySpatialExtractor()
        image=np.zeros((4,4,3),np.uint8)
        mask=np.ones((4,4),bool)
        with patch.object(model,'_forward_tokens',return_value=(None,(2,2))):
            with self.assertRaisesRegex(RuntimeError,'spatial tokens'):
                model.extract_features(image,mask)
            with self.assertRaisesRegex(RuntimeError,'spatial tokens'):
                model.extract_features_batch(image,[mask])

    def test_thin_object_is_not_eroded(self):
        from src import region_extractors as regions
        labels=np.zeros((30,30),np.uint8)
        labels[3:23,4:6]=1
        with patch.object(regions,'DATASET_TYPE','voc'), patch.object(regions,'DATASET_CLASSES',['object']), \
             patch.object(regions,'BUILD_SPLIT_CONNECTED_COMPONENTS',True):
            objects=[r for r in regions._extract_class_regions(labels,1) if r['cls_id']==1]
        self.assertEqual(len(objects),1)
        np.testing.assert_array_equal(objects[0]['mask'],labels==1)

    def test_reject_removed_dual_space(self):
        import json
        from src.feature_signature import validate_feature_signature
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'feature_signature.json').write_text(json.dumps({'aggregation':'dual'}))
            with self.assertRaisesRegex(RuntimeError,'aggregation'):
                validate_feature_signature(directory,{})

    def test_pipeline_recipe_has_only_supported_generation_flags(self):
        from configs.experiment import PSEUDO, cli_options
        from tools.generate_pseudo_masks_sam3 import parse_args
        args=parse_args(cli_options(PSEUDO))
        for key,value in PSEUDO.items():
            self.assertEqual(getattr(args,key),value)

    def test_shared_stage_preserves_failure_and_resumes_completed_output(self):
        from project_utils.experiment import stage
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def fail(attempt):
                (attempt/'partial').write_text('partial')
                raise RuntimeError('failed')
            with self.assertRaises(RuntimeError):
                stage(root,'build',False,fail)
            def finish(attempt):
                result=attempt/'complete'; result.write_text('complete')
                return {'path':str(result)},[result]
            result=stage(root,'build',True,finish)
            self.assertIn('attempt_002',result['path'])
            self.assertEqual(stage(root,'build',True,fail),result)
            Path(result['path']).write_text('modified output')
            with self.assertRaises(ValueError):
                stage(root,'build',True,fail)
