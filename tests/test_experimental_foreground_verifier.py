import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools import experimental_foreground_verifier as v


class VerifierTests(unittest.TestCase):
    def test_selection_training_only_deterministic(self):
        plan={str(i):[dict(cls='c'+str(i))] for i in range(20)}
        plan['held']=[dict(cls='c0')]
        split=dict(train_images=[str(i) for i in range(20)],validation_images=['held'])
        self.assertNotIn('held',v.choose_images(plan,split))
        self.assertEqual(v.choose_images(plan,split),v.choose_images(plan,split))
        with self.assertRaises(ValueError):v.choose_images(plan,dict(train_images=['held'],validation_images=['held']))

    def test_cluster_deterministic_unit_and_insufficient(self):
        x=np.array([[1,0],[.9,.1],[-1,0],[-.9,-.1]],np.float32)
        c=v.cluster(x,2)
        np.testing.assert_allclose(np.linalg.norm(c,axis=1),1,atol=1e-6)
        np.testing.assert_array_equal(c,v.cluster(x,2))
        with self.assertRaises(ValueError):v.cluster(x,5)
        with self.assertRaises(ValueError):v.unit(np.array([[np.nan,0]]))
        with self.assertRaises(ValueError):v.unit(np.zeros((1,2)))

    def test_cache_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages=v.ab.Stages(Path(tmp)/'evidence',True)
            def action(attempt):
                p=attempt/'vectors.npz';np.savez_compressed(p,bg1=np.zeros((20,72,72)))
                return dict(file=str(p)),[p]
            stages.run('one',action)
            p=Path(v.read_cache(tmp,'evidence','one'));p.write_bytes(b'changed')
            with self.assertRaises(ValueError):v.read_cache(tmp,'evidence','one')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_logmeanexp_no_count_bias(self):
        import torch
        tokens=torch.tensor([[1.,0.]]).repeat(5184,1)
        fg=np.tile(np.array([1.,0.],np.float32),(20,4,1))
        bg=np.array([[1.,0.]],np.float32)
        a=v.evidence(tokens,fg,bg)
        b=v.evidence(tokens,fg,np.repeat(bg,8,axis=0))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
        torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-6,rtol=0)
        with self.assertRaises(ValueError):v.evidence(torch.zeros_like(tokens),fg,bg)

    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('scipy'),'OpenCV/scipy unavailable locally')
    def test_core_empty_and_void(self):
        result=v.core_indices(np.full((40,40),255,np.uint8))
        self.assertTrue(all(len(x)==0 for x in result.values()))
        result=v.core_indices(np.ones((40,40),np.uint8))
        self.assertGreater(len(result[1]),0)
        self.assertEqual(len(result[0]),0)


if __name__=='__main__':unittest.main()
