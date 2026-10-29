import unittest
import numpy as np
from tools.bank_postfilter_sweep_core import PERCENTAGES,SEEDS,CASES,trim,summarize,winner,name
from tools.bank_filter_placement_core import trim as oldtrim
from tools.mlp_projector_voc_core import confusion
class Tests(unittest.TestCase):
    def setUp(self):self.x=np.arange(1,401,dtype=np.float32).reshape(100,4);self.ids=np.asarray([str(i) for i in range(100)])
    def test_zero(self):self.assertEqual(trim(self.x,self.ids,0)[0].tolist(),list(range(100)))
    def test_counts(self):
        for p in PERCENTAGES:self.assertEqual(trim(self.x,self.ids,p)[1]['dropped'],p)
    def test_background(self):
        for p in PERCENTAGES:self.assertEqual(trim(self.x,self.ids,p,True)[1]['dropped'],0)
    def test_floor(self):self.assertEqual(trim(self.x[:7],self.ids[:7],15)[1]['dropped'],1)
    def test_old_exact(self):
        for p in [0,5,10]:
            k,r=trim(self.x,self.ids,p);ko,ro=oldtrim(self.x,self.ids,str(p/100));self.assertTrue(np.array_equal(k,ko));self.assertEqual(r['dropped_ids'],ro['dropped_ids'])
    def test_nested(self):
        prev=set(self.ids)
        for p in PERCENTAGES:
            k,_=trim(self.x,self.ids,p);now=set(self.ids[k]);self.assertTrue(now<=prev);prev=now
    def test_source_order(self):
        k,_=trim(self.x,self.ids,20);self.assertTrue(np.all(np.diff(k)>0))
    def test_dtype(self):
        with self.assertRaises(ValueError):trim(self.x.astype(np.float64),self.ids,5)
    def test_nan(self):
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):trim(x,self.ids,5)
    def test_duplicates(self):
        with self.assertRaises(ValueError):trim(self.x,np.repeat('a',100),5)
    def test_unregistered(self):
        for p in [2,7,35,.1,True]:
            with self.assertRaises(ValueError):trim(self.x,self.ids,p)
    def test_ties_match_original(self):
        x=np.ones((40,4),np.float32);ids=self.ids[:40];a,ra=trim(x,ids,10);b,rb=oldtrim(x,ids,'0.1');self.assertEqual(a.tolist(),b.tolist());self.assertEqual(ra['dropped_ids'],rb['dropped_ids'])
    def test_summary_and_tie(self):
        m={f'{name(p)}_{s}_{c}':dict(mIoU=.8+(p==5)*.01) for p in PERCENTAGES for s in SEEDS for c in CASES};q=summarize(m);self.assertEqual(winner(q),5);self.assertAlmostEqual(q['HCR']['p05']['mean_delta10_pp'],1.)
        for p in PERCENTAGES:q['HCR'][name(p)]['mean']=80.
        self.assertEqual(winner(q),0)
    def test_CM_orientation_and_ignore(self):
        cm=confusion(np.array([[9,18,255]]),np.array([[18,9,0]]));self.assertEqual(cm[9,18],1);self.assertEqual(cm[18,9],1);self.assertEqual(cm.sum(),2)
if __name__=='__main__':unittest.main()
