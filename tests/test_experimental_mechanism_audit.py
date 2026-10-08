import importlib.util
import unittest
from types import SimpleNamespace
import numpy as np
from tools import experimental_mechanism_audit as a
from src import retrieval


class AuditTests(unittest.TestCase):
    def test_one_rule_only_and_hole_semantics(self):
        config=dict(min_score=.7,min_mask_area=100,min_island_area=64,max_hole_area=32,max_hole_fraction=.01,nms_iou=.85)
        variants=a.recipes(config)
        self.assertEqual(len(variants),7)
        for name,value in variants.items():
            self.assertEqual(sum(config[k]!=value[k] for k in config),0 if name=='baseline' else 1)
        self.assertEqual(variants['no_hole_fraction_cap']['max_hole_fraction'],1)
        self.assertEqual(variants['no_hole_fill']['max_hole_area'],0)

    def test_coverage_is_union_and_ignores_void(self):
        gt=np.array([[0,1,255]])
        counts=a.coverage_counts(gt,[np.ones_like(gt,bool)]*2,np.array([[0,1,0]],bool))
        self.assertEqual(counts['0']['gained'],1)
        self.assertEqual(counts['1']['gained'],0)
        self.assertEqual(sum(v['covered'] for v in counts.values()),2)

    def test_decomposition_matches_real_classifier(self):
        rng=np.random.default_rng(42)
        gallery=rng.normal(size=(31,8)).astype(np.float32)
        query=rng.normal(size=(7,8)).astype(np.float32)
        gallery/=np.linalg.norm(gallery,axis=1,keepdims=True);query/=np.linalg.norm(query,axis=1,keepdims=True)
        labels=np.array(['chair']*10+['sofa']*11+['background']*10)
        ctx=retrieval.build_retrieval_context(gallery,labels)
        sims,ids=retrieval._search(query,gallery,None,30)
        local,center,fused,support=a.decompose(query,ctx,labels,sims,ids,5,.4)
        actual=retrieval.classify_queries(query,gallery,labels,None,ctx,5,'balanced',5,2,.4)
        for row,pred in zip(fused,actual):
            rank=np.argsort(row)[::-1]
            self.assertEqual(ctx.labels[rank[0]],pred.label)
            self.assertAlmostEqual(float(row[rank[0]]),pred.score,places=6)
            self.assertAlmostEqual(float(row[rank[0]]-row[rank[1]]),pred.margin,places=6)

    def test_absent_class_centroid_fallback_recorded(self):
        ctx=SimpleNamespace(labels=('a','b'),centroids=np.eye(2,dtype=np.float32))
        local,center,_,support=a.decompose(np.array([[1,0]],np.float32),ctx,np.array(['a']),
            np.array([[1.]],np.float32),np.array([[0]]),5,.4)
        self.assertEqual(support[0,1],0)
        self.assertEqual(local[0,1],center[0,1])

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV unavailable locally')
    def test_disabling_score_recovers_candidate(self):
        from src.proposal_cleanup import CleanupConfig,clean_proposals
        config=CleanupConfig(min_score=.7,min_mask_area=0,min_island_area=0,max_hole_area=0,max_hole_fraction=0,nms_iou=1)
        mask=np.ones((1,5,5),bool);scores=np.array([.6],np.float32)
        self.assertEqual(len(clean_proposals(mask,scores,config)[0]),0)
        self.assertEqual(len(clean_proposals(mask,scores,CleanupConfig(**a.recipes(config.to_dict())['no_score']))[0]),1)


if __name__=='__main__':unittest.main()
