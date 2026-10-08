import unittest
from tools import experimental_adaptive_rescue as a
from tools import experimental_rescue_mechanisms as old
from tests.test_experimental_overnight_rescue import row
import numpy as np


class AdaptiveTests(unittest.TestCase):
    def test_strong_singleton_weak_requires_votes(self):
        p=a.VARIANTS['hybrid_margin20']
        self.assertEqual(a.rescue_rows([],[row([1,0],margin=.21)],p)[1],1)
        self.assertEqual(a.rescue_rows([],[row([1,0],margin=.15)],p)[1],0)
        result,n=a.rescue_rows([],[row([1,1,0],margin=.15),row([0,1,1],margin=.15)],p)
        self.assertEqual(n,1);np.testing.assert_array_equal(result[0]['mask'],[[0,1,0]])

    def test_joint_requires_both_and_base_gate_still_applies(self):
        p=a.VARIANTS['hybrid_joint']
        self.assertFalse(a.single_allowed(row([1],retr=.96,margin=.15),p))
        self.assertTrue(a.single_allowed(row([1],retr=.96,margin=.21),p))
        self.assertEqual(a.rescue_rows([],[row([1],retr=.96,margin=.21,q=.1)],p)[1],0)

    def test_controls_exact_and_foreground_immutable(self):
        strong=[row([1,0,0,0])];extras=[row([1,1,1,0]),row([0,0,1,1])]
        for method in ('rescue_strict','vote2'):
            x,n=a.rescue_rows(strong,extras,a.VARIANTS[method]);y,m=old.rescue_rows(strong,extras,old.VARIANTS[method])
            self.assertEqual(n,m)
            for u,v in zip(x,y):np.testing.assert_array_equal(u['mask'],v['mask'])
        for p in a.VARIANTS.values():
            x,_=a.rescue_rows(strong,extras,p)
            for r in x[1:]:self.assertFalse((r['mask']&strong[0]['mask']).any())

    def test_named_class_validation_through_original_validator(self):
        cm=np.diag(np.arange(1,22,dtype=np.int64)*100)+1
        iou=cm.diagonal()/(cm.sum(0)+cm.sum(1)-cm.diagonal())
        v=dict(confusion_matrix=cm.tolist(),mIoU=float(iou.mean()),per_class_iou=dict(zip(a.VOC_CLASSES,map(float,iou))))
        ids=list(map(str,range(1449)))
        v.update(evaluated_ids=ids,evaluated_count=1449,eval_limit=0,cleanup=None,
                 proposal_count_before=99,proposal_count_after=99,
                 inference_settings={'INF_PROPOSAL_SCORE_WEIGHT':.2},proposals_dir='same')
        reference=a.old.canonical(v);v['mIoU']+=1.2692839757022512e-7
        current=a.checked_copy(v)[0];ref=a.checked_copy(reference)[0]
        a.old.validate_eval(current,ref,ids,99,.2)
        current['evaluated_count']=1448
        with self.assertRaises(ValueError):a.old.validate_eval(current,ref,ids,99,.2)

    def test_fixed_plan(self):
        plan=a.evaluation_plan()
        self.assertEqual(len([x for batch in plan for x in batch]),24)
        self.assertEqual([batch[0][0] for batch in plan[:3]],['baseline','rescue_strict','vote2'])


if __name__=='__main__':unittest.main()
