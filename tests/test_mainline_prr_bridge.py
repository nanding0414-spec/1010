"""Interface-only tests: original-image queries and protected foreground."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.progressive_retrieval_recovery import ProgressiveRetrievalRecovery

class PRRBridgeTests(unittest.TestCase):
    def processor(self,invalid=False):
        p=ProgressiveRetrievalRecovery.__new__(ProgressiveRetrievalRecovery)
        class Extractor:
            def extract_features_batch(self,rgb,masks):
                self.rgb=rgb;self.masks=masks
                return [np.zeros(2,np.float32) if invalid else np.array([3,4],np.float32) for m in masks]
        p.proposer=object();p.extractor=Extractor();p.gallery=np.eye(2,dtype=np.float32)
        p.labels=['background','sofa'];p.index=None;p.context=object();p.mapping={0:'background',18:'sofa'}
        return p

    def test_original_image_vectors_same_gallery_and_semantic_class(self):
        p=self.processor();rgb=np.zeros((10,12,3),np.uint8);base=np.zeros((10,12),np.uint8);base[:2,:2]=9
        masks=np.ones((1,10,12),bool);candidate=dict(masks=masks,scores=np.array([.8],np.float32),candidate_ids=np.array(['R3:i:09:1']),source_classes=np.array([9]))
        row=SimpleNamespace(label='sofa',score=.99,margin=.2)
        with patch('tools.oracle_guided_r.decode_image',return_value=(candidate,{})),patch('src.retrieval.classify_queries',return_value=[row]) as classify:
            actual,report=p.recover(rgb,base,'i')
        self.assertIs(p.extractor.rgb,rgb)
        np.testing.assert_array_equal(p.extractor.masks[0],masks[0])
        np.testing.assert_allclose(classify.call_args.args[0],np.array([[.6,.8]],np.float32),atol=1e-7)
        self.assertIs(classify.call_args.args[1],p.gallery)
        np.testing.assert_array_equal(actual[:2,:2],base[:2,:2]);self.assertEqual(int(actual[-1,-1]),18)
        self.assertTrue(report['original_image_semantics'])

    def test_invalid_query_rejected_not_silently_dropped(self):
        p=self.processor(True);candidate=dict(masks=np.ones((1,10,12),bool),scores=np.array([.8],np.float32),candidate_ids=np.array(['R3:i:09:1']),source_classes=np.array([9]))
        with patch('tools.oracle_guided_r.decode_image',return_value=(candidate,{})):
            with self.assertRaisesRegex(ValueError,'Invalid PRR query'):p.recover(np.zeros((10,12,3),np.uint8),np.zeros((10,12),np.uint8),'i')

if __name__=='__main__':unittest.main()
