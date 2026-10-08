import tempfile
from pathlib import Path
import unittest
import numpy as np
from tools import experimental_error_oracles as e


class OracleTests(unittest.TestCase):
    def test_ignore_and_invalid(self):
        cm=e.confusion(np.array([[1,255,0]]),np.array([[1,2,1]]))
        self.assertEqual(cm.sum(),2)
        with self.assertRaises(ValueError):e.confusion(np.array([[22]]),np.array([[0]]))
        with self.assertRaises(ValueError):e.confusion(np.array([[0]]),np.array([[255]]))

    def test_majority_includes_background_ignores_void(self):
        gt=np.array([[0,0,1,255]])
        self.assertEqual(e.majority(np.ones_like(gt,bool),gt),0)
        self.assertEqual(e.majority(gt==255,gt),0)

    def test_pixel_oracle_cannot_invent_foreground(self):
        gt=np.array([[1,2,0]])
        masks=[np.ones_like(gt,bool)]
        pred=e.compose(masks,[1],gt.shape)
        actual=e.pixel_oracle(masks,[1],gt,pred)
        np.testing.assert_array_equal(actual,[[1,1,0]])

    def test_overlap_class_and_pixel_are_distinct(self):
        gt=np.array([[1,2],[1,2]],np.uint8)
        masks=[np.ones_like(gt,bool),gt==2]
        rows=[dict(mask=m,cls_name=c) for m,c in zip(masks,['one','two'])]
        pred=e.compose(masks,[1,2],gt.shape)
        cm,detail=e.diagnose(gt,masks,[.9,.8],rows,['one','two'],pred)
        self.assertEqual(cm['baseline'].trace(),2)
        self.assertEqual(cm['pixel_selected'].trace(),4)
        self.assertEqual(detail['counts']['foreground_wrong_class'],2)
        self.assertEqual(detail['region_iou'],[.5,1.])

    def test_empty_proposals_and_components(self):
        gt=np.array([[1,0],[0,1]],np.uint8)
        cm,d=e.diagnose(gt,[],[],[],['one'],np.zeros_like(gt))
        self.assertEqual(d['region_iou'],[0.,0.])
        self.assertEqual(cm['best_region_pixel'].trace(),2)

    def test_void_not_boundary(self):
        self.assertFalse(e.boundary_band(np.array([[1,255],[1,255]])).any())

    def test_stage_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=e.ab.Stages(Path(tmp),True);calls=[]
            def action(attempt):
                calls.append(1);p=attempt/'data.json';e.ab.write(p,{'x':1})
                return {'path':str(p)},[p]
            value=stage.run('seed42',action)
            self.assertEqual(stage.run('seed42',action),value)
            self.assertEqual(len(calls),1)
            Path(value['path']).write_text('{}')
            with self.assertRaises(ValueError):stage.run('seed42',action)


if __name__=='__main__':unittest.main()
