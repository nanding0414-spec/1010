import unittest
import numpy as np
from tools import experimental_rescue_evidence as e
from tools import experimental_adaptive_rescue as a
from tests.test_experimental_overnight_rescue import row


class EvidenceTests(unittest.TestCase):
    def test_existing_hybrid_exact(self):
        rng=np.random.default_rng(123)
        for _ in range(20):
            strong=[row(rng.random(40)>.8)]
            extras=[row(rng.random(40)>.7,cls='chair' if i%2 else 'bottle',retr=.91+.08*rng.random(),score=rng.random()) for i in range(12)]
            x,n=e.rescue_rows(strong,extras,e.VARIANTS['hybrid_retrieval95'])
            y,m=a.rescue_rows(strong,extras,a.VARIANTS['hybrid_retrieval95'])
            self.assertEqual(n,m)
            for u,v in zip(x,y):np.testing.assert_array_equal(u['mask'],v['mask'])

    def test_high_only_rejects_weak_even_with_votes(self):
        extras=[row([1,1,0],retr=.94),row([0,1,1],retr=.94)]
        self.assertEqual(e.rescue_rows([],extras,e.VARIANTS['high95_only'])[1],0)
        self.assertEqual(e.rescue_rows([],extras,e.VARIANTS['weak_vote2_only'])[1],1)
        self.assertEqual(e.rescue_rows([],[row([1],retr=.95)],e.VARIANTS['high95_only'])[1],1)
        self.assertEqual(e.rescue_rows([],[row([1],retr=.95)],e.VARIANTS['weak_vote2_only'])[1],0)

    def test_weak_voter_excludes_high_support(self):
        extras=[row([1,1,0],retr=.91,score=.99),row([0,1,1],retr=.96,score=.9)]
        x,_=e.rescue_rows([],extras,e.VARIANTS['hybrid_retrieval95'])
        y,_=e.rescue_rows([],extras,e.VARIANTS['hybrid_weak_voters'])
        self.assertEqual(len(x),2);self.assertEqual(len(y),1)
        np.testing.assert_array_equal(y[0]['mask'],[[0,1,1]])

    def test_strong_first_prevents_cross_class_weak_capture(self):
        extras=[row([1,1,0],retr=.91,score=.99),row([0,1,1],retr=.92,score=.98),
                row([0,1,0],cls='bottle',retr=.96,score=.9)]
        x,_=e.rescue_rows([],extras,e.VARIANTS['hybrid_retrieval95'])
        y,_=e.rescue_rows([],extras,e.VARIANTS['hybrid_strong_first'])
        self.assertEqual(x[0]['cls_name'],'chair');self.assertEqual(y[0]['cls_name'],'bottle')

    def test_three_votes_and_empty_pool(self):
        p=e.VARIANTS['hybrid_vote3']
        self.assertEqual(e.rescue_rows([],[],p),([],0))
        extras=[row([1,1,0,0],retr=.91),row([0,1,1,0],retr=.91)]
        self.assertEqual(e.rescue_rows([],extras,p)[1],0)
        extras.append(row([0,1,0,1],retr=.91))
        self.assertEqual(e.rescue_rows([],extras,p)[1],1)

    def test_original_foreground_protected_all_groups(self):
        strong=[row([1,0,0])];extras=[row([1,1,0],retr=.96),row([0,1,1],retr=.91)]
        for p in e.VARIANTS.values():
            rows,_=e.rescue_rows(strong,extras,p)
            for r in rows[1:]:self.assertFalse((r['mask']&strong[0]['mask']).any())

    def test_plan(self):
        plan=e.evaluation_plan()
        self.assertEqual(len(plan),8)
        self.assertEqual([x[0][0] for x in plan[:3]],['baseline','rescue_strict','hybrid_retrieval95'])


if __name__=='__main__':unittest.main()
