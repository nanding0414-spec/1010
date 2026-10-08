"""Isolated RPG experiment tests. CUDA/checkpoint evaluation is not mocked as accuracy."""
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from tools import experimental_rpg_ab as rpg
from tools import experimental_pmfa_ab as ab

try:
    import torch
except ImportError:
    torch=None


class InfrastructureTests(unittest.TestCase):
    def test_parameters_and_signature_isolation(self):
        for kwargs in ({'variant':'A'},{'lambda_base':-1},{'temperature':0},
                       {'core_ratio':0},{'core_threshold':.4},{'temperature':float('nan')}):
            with self.assertRaises(ValueError): rpg.RPG(**kwargs)
        signatures=[rpg.signature({},'rpg',rpg.RPG(variant=v)) for v in ('B','C','D')]
        self.assertEqual(len({str(s) for s in signatures}),3)
        self.assertNotEqual(signatures[2],rpg.signature({},'rpg',rpg.RPG(lambda_base=1)))
        self.assertNotEqual(signatures[2]['format_version'],ab.EXPERIMENT_FORMAT)
        self.assertNotEqual(signatures[2],ab.signature({},'pmfa',ab.PMFA()))
        with self.assertRaises(ValueError): rpg.signature({},'smfa',rpg.RPG())

    def test_reuse_hash_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'weight';file.write_text('original')
            records=ab.identities([file]);rpg.verify_records(records,'test')
            file.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'): rpg.verify_records(records,'test')

    def test_merged_artifact_records_ignore_order_but_keep_hash_checks(self):
        # Exercise both the permanent fix and the no-source-change recovery shim.
        script=(rpg.ROOT/'tools/experimental_rpg_resume.sh').read_text(encoding='utf-8')
        source=script.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        namespace={'__name__':'recovery_test'}
        exec(compile(source,'experimental_rpg_resume.sh','exec'),namespace)
        for verify in (rpg.verify_records,namespace['verify_records']):
            with self.subTest(verifier=verify),tempfile.TemporaryDirectory() as directory:
                a=Path(directory)/'a';z=Path(directory)/'z'
                a.write_text('a');z.write_text('z')
                records=ab.identities([z])+ab.identities([a])
                saved=[dict(r) for r in records]
                verify(records,'merged');verify(records+records[:1],'duplicate')
                self.assertEqual(records,saved)
                with self.assertRaisesRegex(ValueError,'conflicting'):
                    verify(records+[dict(records[0],sha256='wrong')],'conflict')
                z.write_text('edited')
                with self.assertRaisesRegex(ValueError,'changed'):
                    verify(records,'modified')
                z.unlink()
                with self.assertRaisesRegex(ValueError,'missing'):
                    verify(records,'missing')

    def test_recovery_restores_only_exact_historical_source(self):
        import ast
        script=(rpg.ROOT/'tools/experimental_rpg_resume.sh').read_text(encoding='utf-8')
        source=script.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        namespace={'__name__':'recovery_test'}
        exec(compile(source,'experimental_rpg_resume.sh','exec'),namespace)
        restore=namespace['restore_known_verifier_edit']
        current=(rpg.ROOT/'tools/experimental_rpg_ab.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name=='verify_records')
        lines=current.splitlines(keepends=True)
        old_function=("def verify_records(records, label):\n"
                      "    if ab.identities([r['path'] for r in records]) != records:\n"
                      "        raise ValueError(f'{label} files changed or missing')\n")
        historical=(''.join(lines[:node.lineno-1])+old_function+''.join(lines[node.end_lineno:])).encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'tools/experimental_rpg_ab.py'
            target.parent.mkdir();target.write_bytes(historical)
            records=ab.identities([target])
            for variant in ('B','C'):
                ab.write(root/f'experiments/old_{variant}/manifest.json',dict(code=records))
            changed=current.encode('utf-8');target.write_bytes(changed)
            restore(root,'old')
            self.assertEqual(target.read_bytes(),historical)
            backup=next((root/'experiments/old').glob('rpg_source_before_recovery_*.py'))
            self.assertEqual(backup.read_bytes(),changed)
            restore(root,'old')  # Idempotent once source matches.
            target.write_bytes(changed+b'\n# unrelated edit\n')
            before=target.read_bytes()
            with self.assertRaisesRegex(ValueError,'beyond the known'):
                restore(root,'old')
            self.assertEqual(target.read_bytes(),before)

    def test_signature_validator_rejects_other_variants_and_adapter_weights(self):
        from src import feature_signature as fs
        base={key:None for key in fs.REQUIRED_SIGNATURE_KEYS}
        expected=rpg.signature(base,'rpg',rpg.RPG())
        expected['adapter']={'weight_hash':'first'}
        with tempfile.TemporaryDirectory() as directory,\
             patch.object(fs,'REQUIRED_SIGNATURE_KEYS',fs.REQUIRED_SIGNATURE_KEYS|{'experimental_pooling'}):
            fs.save_feature_signature(directory,expected)
            fs.validate_feature_signature(directory,expected)
            for other in (rpg.signature(base,'rpg',rpg.RPG(variant='C')),
                          rpg.signature(base,'rpg',rpg.RPG(temperature=.2)),
                          ab.signature(base,'pmfa',ab.PMFA()),
                          dict(expected,adapter={'weight_hash':'second'})):
                with self.assertRaises(RuntimeError):fs.validate_feature_signature(directory,other)

    def test_variant_does_not_launch_smfa_and_resumes_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); previous=root/'previous'
            context=dict(train_ids=['train'],val_ids=['val'],images={},inputs=[],code=[],
                         selected=dict(pseudo='shared_labels',proposals='shared_proposals'))
            args=types.SimpleNamespace(previous=previous,resume=False,**{k:root/k for k in
                ('dataset_root','image_dir','imagelevel_json','checkpoint','train_list','val_list','gt_dir')})
            base=dict(weights=str(previous/'weights'),metrics=str(previous/'metrics.json'))
            calls=[];fail=[True]
            def launch(job,attempt,env,log,cancel=None):
                self.assertEqual(job['mode'],'rpg');calls.append(job['stage'])
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
                if job['stage']=='build':
                    self.assertEqual(job['pseudo'],'shared_labels')
                    dest=Path(env['OVERRIDE_OUTPUT_ROOT']);dest.mkdir(parents=True);(dest/'data').write_text('raw')
                if job['stage']=='train':
                    self.assertEqual(job['split_file'],str(previous/'weights/training_report.json'))
                    for key in ('bank','weights'):
                        dest=Path(job[key]);dest.mkdir(parents=True);(dest/'data').write_text(key)
                if job['stage']=='eval':
                    if fail[0]: raise RuntimeError('simulated failure')
                    ab.write(job['metrics'],{})
                log.parent.mkdir(parents=True,exist_ok=True);log.write_text('log')
                ab.write(attempt/'worker_report.json',dict(status='complete'))
            artifact_a=root/'a';artifact_z=root/'z'
            artifact_a.write_text('baseline a');artifact_z.write_text('baseline z')
            baseline_records=ab.identities([artifact_z])+ab.identities([artifact_a])
            report=dict(delta_pp=0.,smfa_miou=.5,rpg_miou=.5,per_class_iou={})
            with patch.object(rpg,'ROOT',root),patch.object(ab,'launch',side_effect=launch),\
                 patch.object(ab,'source_identity',return_value=[]),patch.object(rpg,'comparison',return_value=report):
                with self.assertRaisesRegex(RuntimeError,'simulated failure'):
                    rpg.run_variant('test_D',1,rpg.RPG(),args,context,base,baseline_records,threading.Event())
                self.assertEqual(calls,['probe','build','train','eval'])
                self.assertTrue((root/'experiments/test_D/eval/attempt_001/failed.json').is_file())
                self.assertFalse((root/'experiments/test_D/launcher.lock').exists())
                fail[0]=False;args.resume=True
                rpg.run_variant('test_D',1,rpg.RPG(),args,context,base,baseline_records,threading.Event())
                self.assertEqual(calls,['probe','build','train','eval','eval'])
                self.assertTrue((root/'experiments/test_D/eval/attempt_002').is_dir())
                args.resume=False
                with self.assertRaises(ValueError):
                    rpg.run_variant('test_D',1,rpg.RPG(),args,context,base,baseline_records,threading.Event())


@unittest.skipIf(torch is None,'requires PyTorch')
class PoolTests(unittest.TestCase):
    def original(self,t,w):
        if float(w.sum())<1e-6: return t.new_zeros(t.shape[1])
        return torch.nn.functional.normalize((t*w[:,None]).sum(0)/(w.sum()+1e-8),dim=0)

    def test_empty_tiny_zero_identical_and_lambda_one(self):
        for dtype in (torch.float32,torch.float64,torch.float16):
            t=torch.ones(6,4,dtype=dtype)
            for w in (torch.zeros(6,dtype=dtype),torch.tensor([1,1,0,0,0,0],dtype=dtype),torch.ones(6,dtype=dtype)):
                actual=rpg.pool(t,w,rpg.RPG(),self.original)
                torch.testing.assert_close(actual.float(),self.original(t,w).float(),atol=1e-5,rtol=1e-5)
            w=torch.ones(6,dtype=dtype)
            self.assertTrue(torch.equal(rpg.pool(t,w,rpg.RPG(lambda_base=1),self.original),self.original(t,w)))
        self.assertTrue(torch.equal(rpg.pool(torch.zeros(6,4),torch.ones(6),rpg.RPG(),self.original),torch.zeros(4)))

    def test_formula_and_weight_floor(self):
        import torch.nn.functional as F
        t=torch.tensor([[3.,0],[.5,.5],[0,2.],[-1.,.1],[.7,.3]])
        m=torch.tensor([1.,.8,.9,.2,.7])
        for variant in ('B','C','D'):
            p=rpg.RPG(variant=variant);rows=[]
            q0=F.normalize((t*m[:,None]).sum(0)/(m.sum()+1e-8),dim=0,eps=1e-6)
            vn=F.normalize(t,dim=1,eps=1e-6);q=q0
            if variant!='B':
                idx=torch.where(m>=.8)[0]
                q=F.normalize((vn[idx]*m[idx,None]).sum(0)/(m[idx].sum()+1e-8),dim=0,eps=1e-6)
            sim=vn@q;g=torch.sigmoid((sim-sim.median())/.1)
            w=m*(.5+.5*g if variant=='D' else g)
            expected=F.normalize((t*w[:,None]).sum(0)/(w.sum()+1e-8),dim=0)
            actual=rpg.pool(t,m,p,self.original,diagnostics=rows.append)
            torch.testing.assert_close(actual,expected)
            if variant=='D': self.assertGreaterEqual(rows[0]['factor_min'],.5)

    def test_fallback_core_roundoff_and_invalid(self):
        t=torch.eye(5);stats=Counter();rows=[]
        rpg.pool(t,torch.full((5,),.1),rpg.RPG(),self.original,stats,rows.append)
        self.assertEqual(stats['core_all_valid'],1);self.assertEqual(rows[0]['core_tokens'],5)
        w=torch.tensor([1+2**-23, .5, .3, 0., 0.])
        torch.testing.assert_close(rpg.pool(t,w,rpg.RPG(),self.original),rpg.pool(t,w.clamp(0,1),rpg.RPG(),self.original))
        for bad in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):rpg.pool(t,torch.full((5,),bad),rpg.RPG(),self.original)
        with self.assertRaises(ValueError):rpg.pool(t,torch.ones(3),rpg.RPG(),self.original)

    def test_gradient_through_values_only(self):
        t=torch.tensor([[2.,.1],[.2,1.],[.8,.3],[.4,.9]],requires_grad=True)
        w=torch.ones(4)
        value=rpg.pool(t,w,rpg.RPG(),self.original)
        value[0].backward()
        self.assertIsNotNone(t.grad);self.assertTrue(torch.isfinite(t.grad).all())
        self.assertGreater(float(t.grad.abs().sum()),0)

    def test_single_batch_shared_forward_and_diagnostics(self):
        import importlib.util
        import numpy as np
        if not all(importlib.util.find_spec(m) for m in ('cv2','torchvision')):
            self.skipTest('requires cv2/torchvision')
        from src.feature_extractor import BaseFeatureExtractor as Base
        from src.feature_extractor_sam3 import SAM3FeatureExtractor as Sam
        from src import feature_signature as fs
        class Dummy(Base):
            def __init__(self):
                super().__init__(torch.device('cpu'),(1,1),4,'dummy');self.calls=0
            def _preprocess_image(self,rgb):return None,(0,0,2,2)
            def _forward_tokens(self,inp):
                self.calls+=1;return torch.eye(4)[None],(2,2)
        with tempfile.TemporaryDirectory() as directory,\
             patch.object(Base,'_pool_tokens',Base._pool_tokens),\
             patch.object(Base,'extract_features_batch',Base.extract_features_batch),\
             patch.object(Base,'extract_features_for_regions',Base.extract_features_for_regions),\
             patch.object(Sam,'feature_signature',Sam.feature_signature),\
             patch.object(fs,'REQUIRED_SIGNATURE_KEYS',set(fs.REQUIRED_SIGNATURE_KEYS)):
            rpg.install_pooling('rpg',rpg.RPG(),Counter(),dict(attempt=directory,stage='probe'))
            d=Dummy();image=np.zeros((2,2,3),dtype=np.uint8);mask=np.ones((2,2),dtype=bool)
            expected=d.extract_features(image,mask).numpy();d.calls=0
            output=d.extract_features_batch(image,[mask]*33)
            self.assertEqual(d.calls,1)
            for row in output:np.testing.assert_allclose(row,expected,atol=1e-6)
            self.assertEqual(len((Path(directory)/'region_diagnostics.jsonl').read_text().splitlines()),34)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'requires CUDA')
    def test_cuda_matches_cpu(self):
        torch.manual_seed(42);t=torch.randn(17,8);w=torch.rand(17)
        torch.testing.assert_close(rpg.pool(t.cuda(),w.cuda(),rpg.RPG(),self.original).cpu(),
                                   rpg.pool(t,w,rpg.RPG(),self.original),atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
