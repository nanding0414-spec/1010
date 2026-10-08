import unittest
from unittest.mock import patch
import numpy as np
from tools import experimental_error_attribution as a


class AttributionTests(unittest.TestCase):
    def test_scoped_mail_exception_and_restore(self):
        helper=str((a.ROOT/'project_utils/experiment_notify.py').resolve())
        rows=[{'path':helper},{'path':str(a.ROOT/'src/unreviewed.py')}]
        with patch.object(a.ab,'source_identity',return_value=rows) as source:
            def inspect(*args):
                self.assertEqual(a.ab.source_identity(),rows[1:])
                raise ValueError('Other production code rejected')
            with patch.object(a.old,'checked_history',side_effect=inspect):
                with self.assertRaisesRegex(ValueError,'Other production'):
                    a.checked_history(None,None)
            self.assertIs(a.ab.source_identity,source)
            self.assertEqual(a.ab.source_identity(),rows)

    def test_changed_mail_helper_rejected(self):
        with patch.object(a,'NOTIFY_SHA256','wrong'),patch.object(a.old,'checked_history') as history:
            with self.assertRaisesRegex(ValueError,'Email helper differs'):
                a.checked_history(None,None)
            history.assert_not_called()

    def test_split_oracles_disjoint(self):
        gt=np.array([[0,1,1,255]],np.uint8);pred=np.array([[1,2,0,2]],np.uint8)
        rows=[dict(mask=np.ones_like(gt,bool),cls_name='one')]
        out=a.split_oracles(gt,pred,rows,['one','two'])
        np.testing.assert_array_equal(out['background_clear'],[[0,2,0,2]])
        np.testing.assert_array_equal(out['foreground_conflict'],[[1,1,0,2]])
        np.testing.assert_array_equal(out['foreground_recover'],[[1,2,1,2]])
        np.testing.assert_array_equal(out['pixel_combined'],[[0,1,1,2]])

    def test_first_loss_partition(self):
        gt=np.ones((1,7),np.uint8);pred=np.zeros_like(gt)
        masks=[np.eye(7,dtype=bool)[i:i+1] for i in range(2,7)]
        raw=[np.array([[0,1,1,1,1,1,1]],bool)]
        predictions=[dict(label='background',margin=1),dict(label='c1',margin=0),
                     dict(label='c1',margin=1),dict(label='c1',margin=1)]
        rows=[dict(mask=masks[-1],cls_name='c1')]
        out=a.attribute(gt,pred,raw,masks,[1,2,3,4],predictions,.5,rows,[f'c{i}' for i in range(1,21)])
        self.assertEqual(out['c1']['total_fn'],7)
        self.assertTrue(all(out['c1'][k]==1 for k in a.LOSS_NAMES))

    def test_ignore_strip_boundary_proxy(self):
        gt=np.array([[1,255,0]],np.uint8)
        np.testing.assert_array_equal(a.boundary_band(gt),[[True,False,True]])
        self.assertFalse(a.boundary_band(np.ones((3,3),np.uint8)).any())

    def test_alignment_rejected(self):
        with self.assertRaises(ValueError):
            a.attribute(np.ones((1,1)),np.zeros((1,1)),[],[],[0],[],0,[],[])

    def test_aggregate(self):
        d={'attribution':{'chair':{'total_fn':4}}}
        self.assertEqual(a.aggregate_attribution({'a':d,'b':d}),{'chair':{'total_fn':8}})


if __name__=='__main__':unittest.main()
