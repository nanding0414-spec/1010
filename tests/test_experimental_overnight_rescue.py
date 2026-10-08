import unittest
import numpy as np
from tools import experimental_overnight_rescue as r


def row(mask,cls='chair',score=.9,q=.8,margin=.2,retr=.95):
    return dict(mask=np.array([mask],bool),cls_name=cls,score=score,proposal_score=q,retrieval_margin=margin,retrieval_score=retr)


class RescueTests(unittest.TestCase):
    def test_original_foreground_immutable(self):
        strong=[row([1,1,0,0])];extra=[row([0,1,1,0],cls='sofa',score=1)]
        result,n=r.rescue_rows(strong,extra,r.RECIPE['rescue'])
        self.assertEqual(n,1)
        np.testing.assert_array_equal(result[0]['mask'],strong[0]['mask'])
        np.testing.assert_array_equal(result[1]['mask'],[[0,0,1,0]])
        np.testing.assert_array_equal(extra[0]['mask'],[[0,1,1,0]])

    def test_reject_weak_and_background(self):
        extras=[row([1,0],q=.3),row([0,1],margin=.01),row([1,1],cls='background')]
        result,n=r.rescue_rows([],extras,r.RECIPE['rescue'])
        self.assertEqual(n,0);self.assertEqual(result,[])

    def test_strict_is_more_conservative(self):
        extras=[row([1,0],retr=.87,margin=.07)]
        self.assertEqual(r.rescue_rows([],extras,r.RECIPE['rescue'])[1],1)
        self.assertEqual(r.rescue_rows([],extras,r.RECIPE['rescue_strict'])[1],0)

    def test_duplicates_do_not_add_coverage(self):
        extras=[row([1,0]),row([1,0]),row([0,1])]
        result,n=r.rescue_rows([],extras,r.RECIPE['rescue'])
        self.assertEqual(n,2)
        self.assertFalse((result[0]['mask']&result[1]['mask']).any())

    def test_empty_pool(self):
        self.assertEqual(r.rescue_rows([],[],r.RECIPE['rescue']),([],0))


if __name__=='__main__':unittest.main()
