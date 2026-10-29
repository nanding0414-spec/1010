import unittest
import numpy as np
from tools.small_mask_hcr_core import assemble,_nms,ARMS,RULES,summarize

class SmallMaskTests(unittest.TestCase):
    def setUp(self):
        self.names=['background']+['class'+str(i) for i in range(1,21)]
        self.small=np.zeros((30,30),bool);self.small[:10,:20]=True
        self.big=np.zeros((30,30),bool);self.big[10:,:20]=True
    def run_arm(self,arm,masks=None,q=None,flags=None,labels=None,r=None,margin=None):
        masks=np.asarray([self.small] if masks is None else masks);n=len(masks)
        return assemble(masks,[.8]*n if q is None else q,[False]*n if flags is None else flags,
                        [1]*n if labels is None else labels,[.96]*n if r is None else r,[.2]*n if margin is None else margin,self.names,arm)
    def test_baseline_rejects_small(self):self.assertEqual(int(self.run_arm('A_current')[0]['HCR'].sum()),0)
    def test_repair_accepts_high_confidence(self):self.assertEqual(int((self.run_arm('B_gap_hcr')[0]['HCR']!=0).sum()),200)
    def test_direct_keeps_small(self):self.assertEqual(int((self.run_arm('C_area100')[0]['clean']!=0).sum()),200)
    def test_b_proposal_gate(self):self.assertFalse(self.run_arm('B_gap_hcr',q=[.39])[0]['HCR'].any())
    def test_b_retrieval_gate(self):self.assertFalse(self.run_arm('B_gap_hcr',r=[.94])[0]['HCR'].any())
    def test_b_margin_gate(self):self.assertFalse(self.run_arm('B_gap_hcr',margin=[.09])[0]['HCR'].any())
    def test_direct_is_not_high_gate(self):self.assertTrue(self.run_arm('C_area100',r=[.7],margin=[.01])[0]['HCR'].any())
    def test_background_is_not_rescued(self):self.assertFalse(self.run_arm('B_gap_hcr',labels=[0])[0]['HCR'].any())
    def test_raw_small_already_uses_old_hcr(self):
        for a in ARMS:self.assertEqual(int((self.run_arm(a,flags=[True])[0]['HCR']!=0).sum()),200)
    def test_protect_original_hcr(self):
        f,_=self.run_arm('B_gap_hcr',masks=[self.small,self.small],flags=[True,False],labels=[2,1],r=[.97,.96])
        self.assertTrue(np.all(f['HCR'][self.small]==2));self.assertTrue(np.array_equal(f['HCR'],f['before_repair']))
    def test_area_99_not_recovered(self):
        m=np.zeros((30,30),bool);m.flat[:99]=True
        self.assertFalse(self.run_arm('B_gap_hcr',masks=[m])[0]['HCR'].any())
    def test_area_100_recovered(self):
        m=np.zeros((30,30),bool);m.flat[:100]=True
        self.assertTrue(self.run_arm('B_gap_hcr',masks=[m])[0]['HCR'].any())
    def test_area_300_is_normal_strong(self):
        m=np.zeros((30,30),bool);m.flat[:300]=True
        f,t=self.run_arm('B_gap_hcr',masks=[m]);self.assertEqual(t['total_small_clean'],0);self.assertTrue(f['clean'].any())
    def test_float32_decimal_boundary_is_not_epsilon_relaxed(self):
        f,_=self.run_arm('B_gap_hcr',r=[np.float32(.95)]);self.assertFalse(f['HCR'].any())
    def test_stable_tie_original_order(self):
        f,_=self.run_arm('B_gap_hcr',masks=[self.small,self.small],labels=[2,1]);self.assertTrue(np.all(f['HCR'][self.small]==2))
    def test_no_restore_large_nms_reject(self):
        f,t=self.run_arm('B_gap_hcr',masks=[self.big,self.big]);self.assertEqual(t['repair_trace'],[]);self.assertEqual(t['clean_kept'],[0])
    def test_owner_is_pool_index(self):
        f,_=self.run_arm('B_gap_hcr',masks=[self.big,self.small]);self.assertTrue(np.all(f['owner_HCR'][self.small]==1))
    def test_inputs_not_mutated(self):
        masks=np.asarray([self.small,self.big]);old=masks.copy();self.run_arm('B_gap_hcr',masks=masks);np.testing.assert_array_equal(masks,old)
    def test_empty_pool(self):
        f,_=self.run_arm('B_gap_hcr',masks=np.zeros((0,30,30),bool));self.assertFalse(f['HCR'].any())
    def test_novel_reject(self):
        almost=self.small.copy();almost.flat[0]=False
        f,t=self.run_arm('B_gap_hcr',masks=[almost,self.small],flags=[True,False],labels=[2,1],r=[.98,.96]);self.assertEqual(t['repair_trace'][0]['reason'],'novel')
    def test_A_C_match_actual_production_function_bodies(self):
        import ast,typing
        from pathlib import Path
        from types import SimpleNamespace
        path=Path(__file__).resolve().parents[1]/'src/inference.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        names={'mask_iou','nms_masks','nms_masks_per_class','predict_instances','rasterize_instances'}
        scope=dict(np=np,List=typing.List,Dict=typing.Dict,Any=typing.Any,DATASET_CLASSES=self.names[1:],INF_RETRIEVAL_MODE='balanced',INF_CLASS_TOPM=5,INF_RETRIEVAL_SEARCH_MULTIPLIER=2,INF_CENTROID_WEIGHT=.2,INF_RETRIEVAL_MARGIN_THR=0.,INF_PROPOSAL_SCORE_WEIGHT=.2)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'),scope)
        class Extractor:
            def extract_features_batch(self,rgb,masks):return [np.asarray([1.,0.]) for _ in masks]
        from src.semantic_rescue import HCRConfig
        rng=np.random.default_rng(42)
        for _ in range(10):
            masks=[]
            for i in range(12):
                mask=np.zeros((40,40),bool);h,w=rng.integers(8,28,size=2);y,x=rng.integers(0,12,size=2);mask[y:y+h,x:x+w]=True;masks.append(mask)
            qs=rng.uniform(.3,1,12);flags=rng.random(12)>.6;labels=rng.integers(0,5,12);scores=rng.uniform(.8,1,12);margins=rng.uniform(0,.3,12)
            scope['classify_queries']=lambda **kw:[SimpleNamespace(label=self.names[int(c)],score=float(s),margin=float(m)) for c,s,m in zip(labels,scores,margins)]
            for arm,minimum in [('A_current',300),('C_area100',100)]:
                scope['MIN_MASK_AREA']=minimum
                rows=scope['predict_instances'](np.zeros((40,40,3),np.uint8),Extractor(),all_feats=np.eye(2),all_labels=['class1','background'],precomputed_masks=masks,precomputed_scores=qs,retrieval_context=object(),strict_errors=True,rescue_flags=flags.tolist(),rescue_config=HCRConfig())
                expected=scope['rasterize_instances'](rows,40,40);actual,_=assemble(masks,qs,flags,labels,scores,margins,self.names,arm)
                np.testing.assert_array_equal(actual['HCR'],expected)

if __name__=='__main__':unittest.main()
