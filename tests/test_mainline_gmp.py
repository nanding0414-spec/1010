import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from src.hard_center_gmp import select_centers, uniform_gmp, describe
from src.feature_signature import save_feature_signature, validate_feature_signature


class MainlineGMPTests(unittest.TestCase):
    def test_centers_match_historical_geometry(self):
        from tools.experimental_mpsa_ab import selection
        rng = np.random.default_rng(42)
        for shape in ((9,11), (375,500), (1,1)):
            for mask in (np.zeros(shape,bool), np.ones(shape,bool), rng.random(shape)>.95):
                self.assertTrue(np.array_equal(select_centers(mask)[0], selection(mask,(0,0))[0]))
        tiny = np.zeros((375,500),bool); tiny[1,1]=True
        self.assertEqual(select_centers(tiny)[1]['support'], 1)

    def test_gmp_matches_formula_and_historical_operation_order(self):
        from tools.experimental_gmp_ab import solve
        torch.manual_seed(42)
        for n,d in ((1,8),(5,8),(12,8)):
            x=torch.randn(n,d)
            value=uniform_gmp(x,np.arange(n))
            legacy,_=solve(x,torch.ones(n),.1)
            self.assertTrue(torch.equal(value,legacy))
            mu=x.mean(0); s=x.T@x/n
            ref=F.normalize(torch.linalg.solve(s+.1*torch.trace(s)*torch.eye(d),mu),dim=0)
            self.assertTrue(torch.allclose(value,ref,atol=2e-6))
        self.assertTrue(torch.equal(uniform_gmp(torch.ones(3,8),[]),torch.zeros(8)))
        self.assertTrue(torch.equal(uniform_gmp(torch.zeros(3,8),[0]),torch.zeros(8)))

    def test_invalid_inputs_raise(self):
        for mask in (np.array([[np.nan]]), np.array([[2]]),np.zeros((0,2))):
            with self.assertRaises(ValueError):select_centers(mask)
        for rho in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):uniform_gmp(torch.ones(2,4),[0],rho)
        with self.assertRaises(ValueError):uniform_gmp(torch.full((2,4),float('nan')),[0])
        with self.assertRaises(ValueError):uniform_gmp(torch.ones(2,4),[2])

    def test_signature_rejects_algorithm_parameters_checkpoint_and_experiments(self):
        from src.feature_signature import REQUIRED_SIGNATURE_KEYS
        base={k:None for k in REQUIRED_SIGNATURE_KEYS}
        base.update(format_version=3,aggregation=describe(),checkpoint_sha256='a',precision_policy='standard_vit_mlp_fp32_v1')
        with tempfile.TemporaryDirectory() as folder:
            save_feature_signature(folder,base);validate_feature_signature(folder,base)
            for changes in ({'aggregation':None},{'aggregation':describe(.2)},
                            {'checkpoint_sha256':'b'},{'experimental_pooling':{'mode':'other'}}):
                with self.assertRaises(RuntimeError):validate_feature_signature(folder,dict(base,**changes))

    def test_public_paths_share_forward_and_preserve_zero(self):
        from src.feature_extractor_sam3 import SAM3FeatureExtractor
        model=SAM3FeatureExtractor.__new__(SAM3FeatureExtractor)
        model.pooling='hard_center_gmp';model.gmp_rho=.1;model.resolution=1008;model.embed_dim=8
        tokens=torch.randn(1,5184,8);calls=[]
        model._preprocess_image=lambda rgb:(None,None)
        def forward(_):calls.append(1);return tokens,(72,72)
        model._forward_tokens=forward
        rgb=np.zeros((19,23,3),np.uint8);mask=np.ones(rgb.shape[:2],bool);empty=np.zeros_like(mask)
        values=model.extract_features_batch(rgb,[mask,empty])
        self.assertEqual(len(calls),1)
        self.assertTrue(np.array_equal(values[0],model.extract_features(rgb,mask).numpy()))
        self.assertTrue(np.array_equal(values[1],np.zeros(8)))


if __name__=='__main__':unittest.main()
