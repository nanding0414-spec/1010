import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1] / 'experiments/recovery_tools/resume_rescue_fp32_v2.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def metric(self):
        cm = np.diag(np.arange(1,22,dtype=np.int64)*100) + 1
        iou = cm.diagonal()/(cm.sum(0)+cm.sum(1)-cm.diagonal())
        return dict(confusion_matrix=cm.tolist(),mIoU=float(iou.mean()),
                    per_class_iou=dict(zip(r.VOC_CLASSES,map(float,iou))))

    def test_small_roundoff_and_no_mutation(self):
        v=self.metric();v['mIoU']+=1.2692839757022512e-7
        v['per_class_iou']['background']+=1.389106774052351e-7
        original=repr(v);result,_=r.checked_copy(v)
        self.assertEqual(repr(v),original)
        self.assertLess(result['mIoU'],v['mIoU'])

    def test_sorted_dictionary_preserves_named_class_alignment(self):
        import json
        v=self.metric()
        ordered=r.checked_copy(v)[0]
        sorted_value=r.checked_copy(json.loads(json.dumps(v,sort_keys=True)))[0]
        self.assertEqual(ordered,sorted_value)

    def test_swapped_class_values_rejected(self):
        v=self.metric();d=v['per_class_iou']
        d['background'],d['tvmonitor']=d['tvmonitor'],d['background']
        with self.assertRaises(ValueError):r.checked_copy(v)

    def test_large_error_rejected(self):
        v=self.metric();v['mIoU']+=1e-5
        with self.assertRaises(ValueError):r.checked_copy(v)

    def test_class_errors_cannot_cancel(self):
        v=self.metric();v['per_class_iou']['background']+=1e-4;v['per_class_iou']['aeroplane']-=1e-4
        with self.assertRaises(ValueError):r.checked_copy(v)

    def test_invalid_matrix_and_nan_rejected(self):
        v=self.metric();v['confusion_matrix'][0][0]=-1
        with self.assertRaises(ValueError):r.checked_copy(v)
        v=self.metric();v['mIoU']=float('nan')
        with self.assertRaises(ValueError):r.checked_copy(v)


if __name__=='__main__':unittest.main()
