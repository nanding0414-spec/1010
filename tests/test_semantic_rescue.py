import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.semantic_rescue import HCRConfig, rescue_instances, prepare_pool, RescueSource


def row(mask,cls='chair',score=.9,q=.8,margin=.2,retr=.95):
    return dict(mask=np.array([mask],bool),cls_name=cls,score=score,proposal_score=q,
                retrieval_margin=margin,retrieval_score=retr)


class HCRTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'tools/experimental_rescue_evidence.py').exists(),
                         'Historical experiment reference not installed')
    def test_matches_validated_high95(self):
        from tools.experimental_rescue_evidence import rescue_rows, VARIANTS
        rng=np.random.default_rng(42)
        for _ in range(40):
            strong=[row(rng.random(70)>.7)]
            extras=[row(rng.random(70)>.8,cls='chair' if i%2 else 'bottle',
                        q=float(rng.random()),retr=float(rng.random()),margin=float(rng.random()),score=float(rng.random())) for i in range(15)]
            actual=rescue_instances(strong,extras,HCRConfig(),['chair','bottle'])
            expected,_=rescue_rows(strong,extras,VARIANTS['high95_only'])
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a['mask'],b['mask'])

    def test_foreground_and_inputs_unchanged(self):
        original=row([1,1,0]);extra=row([0,1,1],cls='bottle',score=1)
        out=rescue_instances([original],[extra],HCRConfig(),['chair','bottle'])
        np.testing.assert_array_equal(out[1]['mask'],[[0,0,1]])
        np.testing.assert_array_equal(extra['mask'],[[0,1,1]])
        self.assertIs(out[0],original)

    def test_thresholds_background_zero_and_nonfinite(self):
        p=HCRConfig()
        self.assertEqual(rescue_instances([],[],p,['chair']),[])
        for extra in [row([1],retr=.949),row([1],q=.399),row([1],margin=.099),row([1],cls='background'),row([0])]:
            self.assertEqual(rescue_instances([],[extra],p,['chair']),[])
        with self.assertRaises(ValueError):rescue_instances([],[row([1],score=float('nan'))],p,['chair'])
        for kw in [dict(margin=-1),dict(novel_fraction=2),dict(retrieval_score=float('nan'))]:
            with self.assertRaises(ValueError):HCRConfig(**kw)

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV unavailable')
    def test_pool_strong_priority_raw_readonly_and_source(self):
        from src.proposal_cleanup import CleanupConfig
        cfg=CleanupConfig(min_score=.7,min_mask_area=1,min_island_area=1,max_hole_area=0,max_hole_fraction=0,nms_iou=1.)
        raw=np.array([[[1,0,0]],[[0,1,0]],[[0,1,0]]],bool);scores=np.array([.8,.5,.4],np.float32)
        masks,qs,flags=prepare_pool([raw[0]],[.9],raw,scores,cfg)
        self.assertEqual(flags,[False,True]);self.assertEqual(qs[0],.9)
        np.testing.assert_array_equal(raw[0],[[1,0,0]])
        with tempfile.TemporaryDirectory() as d:
            clean=Path(d)/'clean';source=Path(d)/'raw';clean.mkdir();source.mkdir()
            np.savez(source/'x.npz',masks=raw,scores=scores)
            np.savez(clean/'x.npz',masks=raw[:1],scores=[.9])
            (clean/'cleanup_config.json').write_text(json.dumps(dict(config=cfg.to_dict(),input_dir=str(source),errors=0)))
            loader=RescueSource(clean,['x']);out=loader.expand('x',[raw[0]],[.9],1,3)
            self.assertEqual(out[2],flags);self.assertEqual(out[3]['extra_count'],1)
            with self.assertRaises(ValueError):loader.expand('x',[raw[0]],[.9],3,1)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('cv2'),'Torch/OpenCV unavailable')
    def test_predict_flags_survive_invalid_vectors_and_strong_only_nms(self):
        import src.inference as inf
        masks=[np.array([[1,0,0]],bool),np.array([[0,1,0]],bool),np.array([[0,1,1]],bool)]
        class Extractor:
            def extract_features_batch(self,*_):return [np.array([1.,0.]),np.zeros(2),np.array([0.,1.])]
        predictions=[SimpleNamespace(label='chair',score=.96,margin=.2),SimpleNamespace(label='bottle',score=.97,margin=.2)]
        with patch.object(inf,'classify_queries',return_value=predictions), patch.object(inf,'nms_masks_per_class',side_effect=lambda rows,t:rows) as nms:
            rows=inf.predict_instances(np.zeros((1,3,3),np.uint8),Extractor(),all_feats=np.eye(2),all_labels=['chair','bottle'],
                precomputed_masks=masks,precomputed_scores=[.9,.5,.5],retrieval_context=object(),strict_errors=True,
                rescue_flags=[False,True,True],rescue_config=HCRConfig())
            self.assertEqual(len(nms.call_args.args[0]),1);self.assertEqual(len(rows),2)
            pred=inf.rasterize_instances(rows,1,3)
            self.assertEqual(int(pred[0,0]),inf.DATASET_CLASSES.index('chair')+1)
            self.assertEqual(int(pred[0,2]),inf.DATASET_CLASSES.index('bottle')+1)


if __name__=='__main__':unittest.main()
