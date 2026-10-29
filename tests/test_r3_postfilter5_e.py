import unittest
import numpy as np
from tools.r3_postfilter5_e_core import compose,summarize
def row(scores=(.99,.92),margins=(.2,.07),labels=(1,2),ids=('a','b')):
    p=np.full(len(scores),.8,np.float32);r=np.asarray(scores,np.float32)
    return dict(class_ids=np.asarray(labels,np.int16),retrieval_scores=r,margins=np.asarray(margins,np.float32),proposal_scores=p,fused_scores=.8*r+.2*p,candidate_ids=np.asarray(ids),source_classes=np.ones(len(scores),np.int16))
class ComboTests(unittest.TestCase):
    def runrow(self,q=None,base=None,masks=None):
        return compose(np.zeros((1,4),np.uint8) if base is None else base,np.asarray([[[1,1,0,0]],[[0,0,1,1]]] if masks is None else masks,bool),row() if q is None else q,'img')
    def test_strict_then_loose(self):
        d,r=self.runrow();self.assertEqual(d['strict'].tolist(),[[1,1,0,0]]);self.assertEqual(d['E'].tolist(),[[1,1,2,2]])
    def test_protect_base(self):
        d,_=self.runrow(base=np.array([[3,0,0,0]],np.uint8));self.assertEqual(d['E'][0,0],3)
    def test_strict_precedence(self):
        d,_=self.runrow(masks=[[[1,1,0,0]],[[1,1,1,1]]]);self.assertEqual(d['E'].tolist(),[[1,1,2,2]])
    def test_no_double_accept(self):
        _,r=self.runrow();self.assertEqual([x['candidate_id'] for x in r['loose_accepted']],['b'])
    def test_background_cannot_fill(self):
        d,_=self.runrow(q=row(labels=(0,0)));self.assertFalse(d['E'].any())
    def test_live_zero_rejected(self):
        d,r=self.runrow(masks=[[[1,1,0,0]],[[1,1,0,0]]]);self.assertEqual(len(r['loose_accepted']),0)
    def test_float32_threshold_no_epsilon(self):
        d,_=self.runrow(q=row(scores=(.9,.91),margins=(.07,.07)));self.assertEqual(d['E'].tolist(),[[0,0,2,2]])
    def test_proposal_gate(self):
        q=row();q['proposal_scores'][:]=.1;d,_=self.runrow(q);self.assertFalse(d['E'].any())
    def test_margin_gate(self):
        d,_=self.runrow(q=row(margins=(.01,.01)));self.assertFalse(d['E'].any())
    def test_owner_exact(self):
        d,_=self.runrow();np.testing.assert_array_equal(d['owner_strict']>=0,d['strict']!=0);np.testing.assert_array_equal(d['owner_E']>=0,d['E']!=d['strict'])
    def test_empty_candidates(self):
        q=row(scores=(),margins=(),labels=(),ids=());d,_=compose(np.zeros((2,3),np.uint8),np.zeros((0,2,3),bool),q,'x');self.assertFalse(d['E'].any())
    def test_summary_paired(self):
        m={f'{a}_{s}':{'mIoU':.8+(i*.001)+(j*.0001)} for i,a in enumerate(['base','strict','E']) for j,s in enumerate([42,123,2026])};r=summarize(m);self.assertAlmostEqual(r['paired']['E-base']['mean_pp'],.2);self.assertEqual(r['paired']['E-base']['positive_seeds'],3)
