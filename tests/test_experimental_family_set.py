import unittest
import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools import experimental_family_set as exp


class SetTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MODUSEG_TEST_REAL_HISTORY')=='1','Actual gallery identity test requires server history')
    def test_actual_gallery_layout_random_control_and_frozen_centers(self):
        root=exp.ROOT/'experiments/multiview_bank_20261008'
        summary=exp.ab.read(root/'summary.json')
        path=Path(summary['results']['view30_clean_42']['metrics']).parent/'job.json'
        job=exp.ab.read(path);job.update(real_bank=summary['results']['real_clean_42']['bank'],
            selection=str(root/'selection.json'),selection_sha256=exp.ab.digest(root/'selection.json'),method='view30_family_set')
        family=exp.member_layout(job)
        random=exp.member_layout(dict(job,method='view30_random_set'))
        self.assertEqual(len(family[0]),55813);self.assertEqual(len(family[5]),37200)
        np.testing.assert_array_equal(family[0],random[0]);np.testing.assert_array_equal(family[5],random[5])
        self.assertEqual(Counter(zip(family[2],family[1],family[3])),Counter(zip(random[2],random[1],random[3])))
        self.assertGreater(random[7]['regrouped_fraction'],.5)
        original=exp.member_layout(dict(job,bank=job['real_bank'],method='real_set'))
        self.assertEqual(len(original[0]),37200);np.testing.assert_array_equal(original[0],family[5])

    def test_formula_view_balance_duplication_and_extremes(self):
        x=np.array([.9,.6,.8]);v=np.array([0,0,1]);t=.05
        expected=t*np.log((np.exp(x[:2]/t).mean()+np.exp(x[2]/t))/2)
        self.assertAlmostEqual(exp.view_logmeanexp(x,v,t),expected)
        self.assertAlmostEqual(exp.view_logmeanexp([.8],[0]),.8)
        self.assertAlmostEqual(exp.view_logmeanexp([.8,.8,.8],[0,1,2]),.8)
        self.assertAlmostEqual(exp.view_logmeanexp([.9,.6,.9,.6,.8],[0,0,0,0,1]),expected)
        self.assertTrue(-1<=exp.view_logmeanexp([-1,1],[0,1],1e-8)<=1)
        for args in (([],[],.05),([float('nan')],[0],.05),([.8],[0],0),([.8],[.5],.05)):
            with self.assertRaises(ValueError):exp.view_logmeanexp(*args)

    def test_random_same_class_view_group_composition_and_background(self):
        groups=np.repeat(np.arange(20),3);labels=np.array(['cat']*30+['dog']*27+['background']*3);v=np.tile([0,1,2],20)
        r=exp.random_groups(groups,labels,v)
        np.testing.assert_array_equal(r,exp.random_groups(groups,labels,v))
        self.assertEqual(Counter(zip(groups,labels,v)),Counter(zip(r,labels,v)))
        np.testing.assert_array_equal(r[-3:],groups[-3:])
        self.assertGreater(float(np.mean(groups!=r)),.5)

    def test_singleton_sets_equivalent_to_flat_scoring(self):
        from src.retrieval import build_retrieval_context,classify_queries
        rng=np.random.default_rng(1);gallery=rng.normal(size=(20,8)).astype(np.float32)
        gallery/=np.linalg.norm(gallery,axis=1,keepdims=True);q=gallery[:3]
        labels=np.array(['cat']*10+['dog']*10);ctx=build_retrieval_context(gallery,labels)
        groups=np.arange(20);members={i:np.array([i]) for i in range(20)}
        a=classify_queries(q,gallery,labels,None,ctx,20)
        b,_=exp.score_sets(q,gallery,np.tile(np.arange(20),(3,1)),groups,np.zeros(20,np.int64),members,labels,ctx)
        for x,y in zip(a,b):
            self.assertEqual(x.label,y.label);self.assertAlmostEqual(x.score,y.score,places=6);self.assertAlmostEqual(x.margin,y.margin,places=6)

    def test_full_set_rescoring_includes_members_outside_shortlist(self):
        from src.retrieval import RetrievalContext
        q=np.array([[1.,0.]],np.float32);gallery=np.array([[1.,0.],[0.,1.],[.5,.866]],np.float32)
        groups=np.array([0,0,1]);members={0:np.array([0,1]),1:np.array([2])};labels=np.array(['cat','cat','dog'])
        ctx=RetrievalContext(('cat','dog'),np.array([[1.,0.],[0.,1.]],np.float32),np.array([2,1]))
        result,diag=exp.score_sets(q,gallery,np.array([[0,2]]),groups,np.array([0,1,0]),members,labels,ctx)
        expected=.6*exp.view_logmeanexp([1.,0.],[0,1])+.4
        self.assertAlmostEqual(result[0].score,expected,places=6);self.assertEqual(diag[0]['candidate_sets'],2)


if __name__=='__main__':unittest.main()
