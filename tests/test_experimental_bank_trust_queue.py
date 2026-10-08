import importlib.util
import unittest
import numpy as np
from tools.experimental_bank_trust_queue import reliability, class_thresholds, supports, altered_predictions, summarize
from src.retrieval import build_retrieval_context, classify_queries


class TrustTests(unittest.TestCase):
    def test_reliability_direction_and_finite(self):
        x=np.array([[.9,.8],[.8,.9],[.8,.8]],np.float32)
        r=reliability(x,np.zeros(3,int))
        self.assertGreater(r[0],.5);self.assertLess(r[1],.5);self.assertEqual(r[2],.5)
        with self.assertRaises(ValueError):reliability([[np.nan,0]],[0])
        with self.assertRaises(ValueError):reliability([[1,0]],[2])

    def test_thresholds_use_own_pseudo_class(self):
        s=np.array([[.2,1],[.8,1],[1,.3],[1,.9]])
        t=class_thresholds(s,[0,0,1,1],['a','b'],.05)
        np.testing.assert_allclose(t,[.23,.33])
        with self.assertRaises(ValueError):class_thresholds(s,[0,0,1,1],['a','b'],.1)

    def test_neutral_formula_and_rejection(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(24,7)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
        q=x[:4];labels=['a']*8+['b']*8+['background']*8;ctx=build_retrieval_context(x,labels)
        common=dict(all_feats=x,all_labels=labels,context=ctx,topk=5,class_topm=3,search_multiplier=2,centroid_weight=.4)
        old=classify_queries(q,faiss_index=None,**common)
        new,_=altered_predictions(q,index=None,trust=np.ones(len(x),np.float32),thresholds=None,**common)
        self.assertEqual([r.label for r in old],[r.label for r in new])
        np.testing.assert_allclose([r.score for r in old],[r.score for r in new],atol=2e-7)
        rejected,details=altered_predictions(q,index=None,trust=None,thresholds=[2,2,2],**common)
        self.assertTrue(all(r.label=='background' for r in rejected))
        self.assertTrue(any(d['rejected'] for d in details))
        empty,_=altered_predictions(q[:0],index=None,trust=None,thresholds=None,**common)
        self.assertEqual(empty,[])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_same_image_exclusion_matches_brute_force(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(14,4)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
        labels=np.array(['a']*7+['b']*7)
        images=np.array(['shared','shared','a2','a3','a4','a5','a6','shared','b1','b2','b3','b4','b5','b6'])
        trust=np.ones(14,np.float32)
        got,weighted,own,names=supports(x,labels,images,trust,device='cpu')
        expected=np.array([[np.sort((x[i]@x.T)[(labels==c)&(images!=images[i])])[-5:].mean() for c in names] for i in range(len(x))])
        np.testing.assert_allclose(got,expected,atol=2e-7)
        np.testing.assert_array_equal(got,weighted)

    def test_incomplete_group_has_no_mean(self):
        s=summarize({'baseline_42':dict(miou=.8)})
        self.assertFalse(s['groups']['baseline']['complete'])
        self.assertIsNone(s['groups']['baseline']['mean'])


if __name__=='__main__':unittest.main()
