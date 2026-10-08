"""GMP numerical contracts and CPU-only orchestration helpers."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from tools import experimental_gmp_ab as gmp

try:
    import torch
except ImportError:
    torch=None


class InfrastructureTests(unittest.TestCase):
    def test_selection_internal_only_and_conservative_tie(self):
        self.assertEqual(gmp.choose(dict(rho1=.8,rho01=.8),.7),'rho1')
        self.assertEqual(gmp.choose(dict(rho1=.8,rho01=.81),.7),'rho01')
        self.assertIsNone(gmp.choose(dict(rho1=.8,rho01=.79),.8))
        with self.assertRaises(ValueError):gmp.choose(dict(rho1=float('nan')),.8)

    def test_signatures_and_invalid_parameters(self):
        for value in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):gmp.GMP(value)
        a=gmp.signature({},'gmp',gmp.GMP(1.));b=gmp.signature({},'gmp',gmp.GMP(.1))
        self.assertNotEqual(a,b)
        self.assertNotEqual(a['format_version'],gmp.ab.EXPERIMENT_FORMAT)
        with self.assertRaises(ValueError):gmp.signature({},'smfa',gmp.GMP())

    def test_resume_image_rechecks_artifacts_and_unsorted_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';z=root/'z';a.write_text('a');z.write_text('z')
            records=gmp.ab.identities([z])+gmp.ab.identities([a])
            gmp.verify(records,'unsorted')
            gmp.ab.write(root/'complete.json',dict(result=dict(file=str(a)),artifacts=records))
            self.assertEqual(gmp.image_cache(root)['file'],str(a))
            z.write_text('changed')
            with self.assertRaises(ValueError):gmp.image_cache(root)

    def test_merge_preserves_exact_members_and_signature(self):
        from src.feature_bank_storage import save_feature_matrix_npz,load_feature_matrix_npz
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';cache=root/'cache';raw=root/'new'
            save_feature_matrix_npz(base/'features/cat/features.npz',np.eye(2,dtype=np.float32),['a_000000','b_000001'])
            gmp.ab.write(base/'meta/class_mapping.json',{'id_to_dirname':{'0':'cat'}})
            for stem,sid,vec in [('a','a_000000',[1,0]),('b','b_000001',[0,1])]:
                folder=cache/stem;folder.mkdir(parents=True);file=folder/'features.npz'
                np.savez(file,features_1=[vec],features_01=[vec],ids=[sid],classes=['cat'])
                gmp.ab.write(folder/'complete.json',dict(result=dict(file=str(file),source_signature=dict(embed_dim=2)),artifacts=gmp.ab.identities([file])))
            gmp.merge(cache,['b','a'],raw,base,1.)
            f,ids=load_feature_matrix_npz(raw/'features/cat/features.npz')
            self.assertEqual(ids.tolist(),['a_000000','b_000001']);np.testing.assert_equal(f,np.eye(2))
            self.assertEqual(gmp.ab.read(raw/'meta/feature_signature.json')['experimental_pooling']['parameters']['rho'],1.)
            with self.assertRaises(ValueError):gmp.merge(cache,['a'],root/'bad',base,.1)

    def test_parallel_failure_cancels_own_waiting_lane(self):
        cancel=threading.Event();barrier=threading.Barrier(2)
        def fail():barrier.wait(3);raise ValueError('failure')
        def peer():barrier.wait(3);self.assertTrue(cancel.wait(3))
        with self.assertRaisesRegex(ValueError,'failure'):gmp.parallel([fail,peer],cancel)


@unittest.skipIf(torch is None,'requires PyTorch')
class SolverTests(unittest.TestCase):
    def test_matches_float64_reference_primal_and_dual(self):
        rng=np.random.default_rng(42)
        for shape in ((17,7),(3,9)):
            f=rng.normal(size=shape);m=rng.uniform(.1,1,shape[0]);p=m/m.sum();energy=np.sum(p*np.sum(f*f,1))
            for rho in (1.,.1):
                z=np.linalg.solve(f.T@(p[:,None]*f)+rho*energy*np.eye(shape[1]),p@f);z/=np.linalg.norm(z)
                actual,info=gmp.solve(torch.tensor(f,dtype=torch.float32),torch.tensor(m,dtype=torch.float32),rho)
                np.testing.assert_allclose(actual.numpy(),z,atol=2e-5,rtol=2e-5)
                self.assertLess(info['relative_residual'],1e-4)

    def test_zero_single_repeated_replication_and_scale(self):
        for f,m in ((torch.zeros(4,3),torch.ones(4)),(torch.ones(4,3),torch.zeros(4))):
            out,_=gmp.solve(f,m,1.);self.assertTrue(torch.equal(out,torch.zeros(3)))
        f=torch.tensor([[1.,0.],[0.,1.],[1.,0.] ]);m=torch.tensor([1.,.5,1.])
        out,_=gmp.solve(f,m,.1)
        for t,w in ((f.repeat(3,1),m.repeat(3)),(f*4,m),(f,m*.5)):
            other,_=gmp.solve(t,w,.1);torch.testing.assert_close(out,other,atol=1e-5,rtol=1e-5)
        out,_=gmp.solve(torch.tensor([[0.,2.]]),torch.ones(1),.1)
        torch.testing.assert_close(out,torch.tensor([0.,1.]))

    def test_invalid_roundoff_and_zero_cancellation(self):
        f=torch.eye(3);m=torch.tensor([1+2**-23,.8,.5])
        a,_=gmp.solve(f,m,.1);b,_=gmp.solve(f,m.clamp(0,1),.1);torch.testing.assert_close(a,b)
        for bad in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):gmp.solve(f,torch.full((3,),bad),.1)
        out,_=gmp.solve(torch.tensor([[1.,0.],[-1.,0.]]),torch.ones(2),.1)
        torch.testing.assert_close(out,torch.zeros(2))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'requires CUDA')
    def test_cpu_cuda_agreement(self):
        torch.manual_seed(42);f=torch.randn(20,8);m=torch.rand(20)
        a,_=gmp.solve(f,m,.1);b,_=gmp.solve(f.cuda(),m.cuda(),.1)
        torch.testing.assert_close(a,b.cpu(),atol=2e-5,rtol=2e-5)


if __name__=='__main__':unittest.main()
