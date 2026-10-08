import unittest
import numpy as np
from tools.experimental_bank_provenance import histogram, keyed_plan, summarize_trace


class ProvenanceTests(unittest.TestCase):
    def test_ignore_and_mixed_region(self):
        r=histogram(np.ones((2,2)),np.array([[0,1],[1,255]],np.uint8))
        self.assertEqual(r['valid_pixels'],3)
        self.assertEqual(r['majority'],1)
        self.assertAlmostEqual(r['purity'],2/3)
        self.assertIsNone(histogram(np.zeros((2,2)),np.zeros((2,2),np.uint8))['majority'])

    def test_bad_geometry_and_labels(self):
        with self.assertRaises(ValueError):histogram(np.ones((2,2)),np.zeros((1,2)))
        with self.assertRaises(ValueError):histogram(np.ones((1,1)),np.array([[22]]))

    def test_class_is_part_of_identity(self):
        p={'image':[dict(cls='cat',id='image_000001',index=4),dict(cls='dog',id='image_000001',index=9)]}
        self.assertEqual(len(keyed_plan(p)),2)
        self.assertEqual(keyed_plan(p)[('cat','image_000001')]['index'],4)
        p['image'].append(p['image'][0])
        with self.assertRaises(ValueError):keyed_plan(p)

    def test_direction_and_purity(self):
        row=dict(purity=1.,truth_majority='cat',spaces={
            'raw':dict(prediction='cat'),
            'mlp':dict(prediction='dog',local_prediction='cat',neighbors=[],decision_neighbors={'dog':[]})})
        result=summarize_trace([row,dict(row,purity=.5)])['counts']
        self.assertEqual(result['candidates'],2)
        self.assertEqual(result['pure_candidates'],1)
        self.assertEqual(result['raw_correct_mlp_wrong'],1)
        self.assertEqual(result['local_correct_fused_wrong'],1)

    def test_decomposition_matches_production_with_absent_class(self):
        from src.retrieval import build_retrieval_context, classify_queries, _search
        from tools.experimental_mechanism_audit import decompose
        x=np.array([[1.,0.],[.98,.2],[-1.,0.]],np.float32)
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        q=np.array([[1.,0.]],np.float32)
        labels=np.array(['cat','cat','dog'])
        ctx=build_retrieval_context(x,labels)
        sims,ids=_search(q,x,None,2)
        local,center,fused,support=decompose(q,ctx,labels,sims,ids,1,.4)
        pred=classify_queries(q,x,labels,None,ctx,1,class_topm=1,search_multiplier=1,centroid_weight=.4)[0]
        self.assertEqual(support[0,1],0)
        self.assertEqual(local[0,1],center[0,1])
        self.assertEqual(pred.label,ctx.labels[np.argmax(fused[0])])
        self.assertEqual(pred.score,float(fused.max()))


if __name__=='__main__':unittest.main()
