import ast,inspect,unittest
import numpy as np
from tools import r3_crossview_control_core as C


def sample(masks=None, A=None, ids=None, scores=None):
    masks=np.asarray(masks if masks is not None else [[[1,1,0,0]],[[0,0,1,1]]],bool)
    n=len(masks)
    return dict(A=np.zeros(masks.shape[1:],np.uint8) if A is None else A, masks=masks,
                candidate_ids=np.asarray(ids or [f'c{i}' for i in range(n)]),labels=np.ones(n,np.int16),
                scores=np.asarray(scores or [.96]*n,np.float32),margins=np.full(n,.2,np.float32),
                proposals=np.full(n,.8,np.float32),accepted_A=set())


class ControlTests(unittest.TestCase):
    def test_preserve_A(self):
        d=sample(A=np.asarray([[2,0,0,0]],np.uint8));r=C.compose({'a':d},2,2)
        self.assertEqual(r['predictions']['a'][0,0],2)
    def test_actual_count_not_eligible_count(self):
        d=sample(masks=[[[1,1,0,0]],[[1,1,0,0]],[[0,0,1,1]]]);r=C.compose({'a':d},2,4)
        self.assertEqual(len(r['eligible']['a']),3);self.assertEqual(len(r['accepted']),2)
    def test_count_shortage_raises(self):
        with self.assertRaisesRegex(ValueError,'Count match impossible'):C.compose({'a':sample()},3,0)
    def test_exclude_A_acceptances(self):
        d=sample();d['accepted_A']={'c0'};r=C.compose({'a':d},1,2)
        self.assertEqual(r['eligible']['a'],['c1'])
    def test_bg_class_cannot_add(self):
        d=sample();d['labels'][0]=0;r=C.compose({'a':d},1,2);self.assertEqual(len(r['accepted']),1)
    def test_global_rank_across_images(self):
        r=C.compose({'a':sample(scores=[.92,.91]),'b':sample(scores=[.99,.98])},1,2)
        self.assertEqual(r['accepted'][0]['image_id'],'b')
    def test_ties_stable_ids(self):
        r=C.compose({'b':sample(ids=['z','a']),'a':sample(ids=['z','a'])},1,2)
        self.assertEqual((r['accepted'][0]['image_id'],r['accepted'][0]['candidate_id']),('a','a'))
    def test_live_novel(self):
        d=sample(masks=[[[1,1,1,1,1,1,1,1,1,0]],[[1,1,1,1,1,1,1,1,1,1]]]);r=C.compose({'a':d},2,9)
        self.assertEqual(r['accepted'][1]['novel_fraction'],.1)
    def test_zero_novel_rejected(self):
        d=sample(masks=[[[1,1]],[[1,1]]]);r=C.compose({'a':d},1,2);self.assertEqual(len(r['accepted']),1)
    def test_fp32_decimal_semantics(self):
        d=sample(scores=[.90,.91]);r=C.compose({'a':d},1,2);self.assertEqual(r['eligible']['a'],['c1'])
    def test_area_closest_prefix(self):self.assertEqual(C.closest_prefix([8,7,20],17),(2,15,2))
    def test_area_tie_shorter(self):self.assertEqual(C.closest_prefix([8,8],12),(1,8,4))
    def test_zero_prefix_allowed(self):self.assertEqual(C.closest_prefix([10],1),(0,0,1))
    def test_not_knapsack(self):self.assertEqual(C.closest_prefix([8,7,20],20),(2,15,5))
    def test_prefix_owner(self):
        d=sample();r=C.compose({'a':d},1,2);p,o=C.prefix(d['A'],r['predictions']['a'],r['owners']['a'],r['ordinals']['a'],1)
        self.assertEqual(int((o>=0).sum()),2);self.assertEqual(int((p>0).sum()),2)
    def test_deterministic(self):
        a=C.compose({'a':sample()},1,2);b=C.compose({'a':sample()},1,2)
        self.assertEqual(a['accepted'],b['accepted']);np.testing.assert_array_equal(a['owners']['a'],b['owners']['a'])
    def test_no_crop_dependency(self):
        d=sample();d['crop_scores']=np.asarray([0,1]);a=C.compose({'a':d},1,2);d['crop_scores'][:]=99
        b=C.compose({'a':d},1,2);self.assertEqual(a['accepted'],b['accepted'])
    def test_no_IO_GT_in_policy(self):
        tree=ast.parse(inspect.getsource(C));names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
        self.assertFalse({'open','Path','gt','torch','Image'}&names)
    def test_quality_denominators(self):
        t=np.zeros((21,21),np.int64);t[1,1]=8;t[0,1]=1;t[2,1]=1;q=C.quality_from_transfer(t,2,2)
        self.assertEqual(q['precision_valid'],.8);self.assertEqual(q['precision_all'],8/12);self.assertEqual(q['net_valid_benefit'],6)
    def test_group_boundaries(self):
        self.assertEqual(C.quality_group(8,10),'GOOD');self.assertEqual(C.quality_group(5,10),'MID');self.assertEqual(C.quality_group(4,10),'BAD');self.assertEqual(C.quality_group(0,0),'UNAVAILABLE_NO_VALID_PIXELS')
    def test_class_cosine_not_probability(self):
        q=dict(retrieval_scores=np.array([.9]),margins=np.array([.1]),class_scores=np.ones((1,21)))
        self.assertAlmostEqual(C.confidence_diagnostic(q,q,0)['class_scores_cosine'],1)


if __name__=='__main__':unittest.main()
