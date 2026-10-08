"""Synthetic protocol tests; real SAM3 and historical replay run on the server."""
from collections import namedtuple
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tools import experimental_mpsa_combo_queue as q


class QueueTests(unittest.TestCase):
    def test_fixed_six_methods_sixteen_new_trainings(self):
        jobs=[x for seed in q.SEEDS for x in q.task_order(seed)]
        self.assertEqual(len(jobs),16)
        self.assertEqual(jobs[:4],[('C',42),('D',42),('E',42),('F',42)])
        self.assertEqual(jobs[4:10],[(g,123) for g in 'ABCDEF'])
        with self.assertRaises(ValueError): q.Recipe('unknown','a'*64)

    def test_signatures_preserve_old_means_and_isolate_combos(self):
        base={'backbone':'sam3','format_version':2}
        for group,n in [('A',1),('B',2)]:
            self.assertEqual(q.signature(base,'combo',q.Recipe(group,'a'*64)),
                             q.mpsa.signature(base,'mpsa',q.mpsa.MPSA(n,'a'*64)))
        signatures=[q.signature(base,'combo',q.Recipe(g,'a'*64)) for g in q.GROUPS]
        self.assertEqual(len({str(s) for s in signatures}),6)
        self.assertNotEqual(signatures[2],q.signature(base,'combo',q.Recipe('C','b'*64)))
        self.assertEqual(q.signature(base,'combo',q.Recipe('SMFA','a'*64))['format_version'],3001)

    def test_time_and_disk_boundaries(self):
        Disk=namedtuple('Disk','total used free')
        with patch.object(q.time,'time',return_value=10),patch.object(q.shutil,'disk_usage',return_value=Disk(100,0,q.FLOOR)):
            q.ensure_resources(11)
            with self.assertRaises(q.Deferred): q.ensure_resources(10)
        with patch.object(q.shutil,'disk_usage',return_value=Disk(100,0,q.FLOOR-1)):
            with self.assertRaises(q.Deferred): q.ensure_resources(float('inf'))

    def test_round_local_failure_continues_and_one_worker_per_gpu(self):
        active=set(); seen=[]; failures=[]; lock=threading.Lock()
        def action(item,gpu):
            with lock:
                self.assertNotIn(gpu,active); active.add(gpu); seen.append(item)
            try:
                time.sleep(.005)
                if item==2: raise q.TaskFailure('isolated')
            finally:
                with lock: active.remove(gpu)
        with patch.object(q,'ensure_resources'):
            q.run_round(list(range(8)),[0,1],action,threading.Event(),100,lambda i,e:failures.append(i))
        self.assertEqual(sorted(seen),list(range(8))); self.assertEqual(failures,[2])

    def test_global_failure_and_cancellation(self):
        cancel=threading.Event()
        with patch.object(q,'ensure_resources'):
            with self.assertRaises(q.GlobalFailure):
                q.run_round([1,2],[0],lambda *_:(_ for _ in ()).throw(q.GlobalFailure('bad signature')),cancel,100,lambda *_:None)
        self.assertTrue(cancel.is_set())
        with patch.object(q,'ensure_resources'),patch.object(q.subprocess,'Popen') as spawn:
            with self.assertRaises(q.GlobalFailure):
                q.launch({},Path('unused'),{},Path('unused'),cancel,100)
            spawn.assert_not_called()

    def test_shared_failure_sets_cancel_and_deadline_starts_nothing(self):
        cancel=threading.Event()
        with patch.object(q,'ensure_resources'):
            with self.assertRaises(q.GlobalFailure):
                q.run_round([1],[0],lambda *_:(_ for _ in ()).throw(q.TaskFailure('extract')),
                    cancel,100,lambda *_:(_ for _ in ()).throw(q.GlobalFailure('shared')))
        self.assertTrue(cancel.is_set())
        with patch.object(q,'ensure_resources',side_effect=q.Deferred('time')),patch.object(q,'worker') as work:
            with self.assertRaises(q.Deferred): q.run_round([1,2],[0,1],work,threading.Event(),100,lambda *_:None)
            work.assert_not_called()

    def test_cache_identity_mask_order_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); file=folder/'vectors.npz'
            identity={'space':'fixed'}; vectors=np.ones((2,1280),np.float32)
            np.savez_compressed(file,D=vectors)
            q.ab.write(folder/'complete.json',dict(file=str(file),identity=identity,mask_hashes=['first','second'],artifacts=q.ab.identities([file])))
            np.testing.assert_array_equal(q.cache_vectors(folder,identity,'D',['first','second']),vectors)
            with self.assertRaises(q.GlobalFailure): q.cache_vectors(folder,identity,'D',['second','first'])
            with self.assertRaises(q.GlobalFailure): q.cache_vectors(folder,{'space':'wrong'},'D',['first','second'])
            file.write_bytes(b'corrupted')
            with self.assertRaises(q.GlobalFailure): q.cache_read(folder,identity)

    def test_summary_partial_and_interactions(self):
        reports={f'{g}_42':dict(candidate_miou=v,delta_pp=100*(v-.85)) for g,v in zip('ABCDEF',[.85,.86,.855,.87,.852,.864])}
        result=q.summarize(reports)
        self.assertFalse(result['groups']['D']['complete'])
        self.assertAlmostEqual(result['contrasts_pp']['42']['gmp_interaction'],.5)
        self.assertAlmostEqual(result['contrasts_pp']['42']['gate_interaction'],.2)
        self.assertIsNone(result['contrasts_pp']['123']['C-A'])

    def test_no_overwrite_new_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            first=q.next_attempt(folder); (first/'partial').write_text('keep')
            second=q.next_attempt(folder)
            self.assertEqual(second.name,'attempt_002'); self.assertEqual((first/'partial').read_text(),'keep')

    def test_worker_launch_failure_categories_and_owned_signal(self):
        class Process:
            pid=123; returncode=1
            def poll(self): return self.returncode
        for category,exception in [('global',q.GlobalFailure),('oom',q.TaskOOM),('local',q.TaskFailure),('deferred',q.Deferred)]:
            with tempfile.TemporaryDirectory() as folder:
                folder=Path(folder); q.ab.write(folder/'queue_worker.json',dict(category=category,error='test'))
                with patch.object(q,'ensure_resources'),patch.object(q.subprocess,'check_output',return_value='0, 0'),\
                     patch.object(q.subprocess,'Popen',return_value=Process()):
                    with self.assertRaises(exception): q.launch({'stage':'eval'},folder,{'CUDA_VISIBLE_DEVICES':'1'},folder/'log',threading.Event(),100)
        class Running:
            pid=777; returncode=None
            def poll(self): return None
            def wait(self,timeout): self.returncode=-15; return -15
        class Cancel:
            def is_set(self): return False
            def wait(self,seconds): return True
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            with patch.object(q,'ensure_resources'),patch.object(q.subprocess,'check_output',return_value='0, 0'),\
                 patch.object(q.subprocess,'Popen',return_value=Running()),patch.object(q,'os',SimpleNamespace(name='posix')) as fake_os:
                from unittest.mock import Mock
                fake_os.killpg=Mock()
                with self.assertRaises(q.GlobalFailure): q.launch({'stage':'train'},folder,{'CUDA_VISIBLE_DEVICES':'0'},folder/'log',Cancel(),100)
                fake_os.killpg.assert_called_once_with(777,q.signal.SIGTERM)

    def test_complete_pipeline_retry_and_resume(self):
        import copy
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory); stack.callback(os.chdir,Path.cwd())
            checkpoint=root/'checkpoint'; checkpoint.write_text('weights')
            options={k:str(root/k) for k in ('dataset_root','image_dir','train_list','val_list','imagelevel_json','gt_dir')}
            options['checkpoint']=str(checkpoint)
            clean=root/'clean'; pseudo=root/'pseudo'; original=root/'original'
            q.ab.write(clean/'cleanup_config.json',{})
            split=dict(train_images=['t'],validation_images=['heldout'])
            source=dict(backbone='sam3',format_version=2,embed_dim=1280)
            sha=q.ab.digest(checkpoint)
            historic=dict(evaluated_ids=['v'],confusion_matrix=np.eye(21,dtype=int).tolist(),cleanup=None,
                proposal_count_before=3,proposal_count_after=3)
            def reference(group,seed):
                path=root/f'old_{group}_{seed}'; sig=q.signature(source,'combo',q.Recipe(group,sha))
                q.ab.write(path/'raw/meta/feature_signature.json',sig)
                q.ab.write(path/'weights/training_report.json',split)
                q.ab.write(path/'metrics.json',historic)
                return dict(raw=str(path/'raw'),weights=str(path/'weights'),bank=str(path/'bank'),metrics=str(path/'metrics.json'))
            smfa={s:reference('SMFA',s) for s in q.SEEDS}; refs={g:reference(g,42) for g in 'AB'}; base=smfa[42]
            prev=root/'previous'; q.ab.write(prev/'manifest.json',dict(previous=str(root/'baseline')))
            old=dict(options=options,selected=dict(pseudo=str(pseudo),proposals=str(original)))
            images={s:str(root/f'{s}.jpg') for s in ('t','v')}
            for p in [*map(Path,images.values()),pseudo/'t.png',clean/'v.npz']:
                p.parent.mkdir(exist_ok=True,parents=True); p.write_bytes(b'input')
            context=dict(inputs=q.ab.identities([*map(Path,images.values()),pseudo/'t.png',clean/'v.npz']),code=[],
                selected=dict(pseudo=str(pseudo),proposals=str(clean)),train_ids=['t'],val_ids=['v'],images=images)
            args=SimpleNamespace(**{k:Path(v) for k,v in options.items()},proposals_dir=clean)
            launched=[]; failed_once=set()
            def launch(job,attempt,env,log,*_,**__):
                launched.append((job['stage'],job['group'],job['seed']))
                if (job['stage'],job['group'],job['seed'])==('train','C',42) and not failed_once:
                    failed_once.add('C'); raise q.TaskFailure('one transient failure')
                log.parent.mkdir(parents=True,exist_ok=True); log.write_text('ok')
                q.ab.write(attempt/'queue_worker.json',dict(status='complete'))
                if job['stage'] in ('probe','extract'):
                    for stem in job['ids']:
                        folder=Path(job['cache'])/('train' if stem=='t' else 'val')/stem
                        q.ab.write(folder/'complete.json',dict(identity=job['cache_identity'][stem],artifacts=[]))
                elif job['stage']=='train':
                    q.ab.write(Path(job['weights'])/'training_report.json',split)
                    q.ab.write(Path(job['bank'])/'meta.json',{})
                    q.ab.write(attempt/'effective_training_argv.json',dict(seed=job['seed']))
                else:
                    q.ab.write(Path(job['metrics']),historic)
                    np.savez_compressed(attempt/'per_image_confusion.npz',ids=['v'],confusions=[historic['confusion_matrix']])
            def merge(_cache,_ids,path,_base,group,sig,_identity): q.ab.write(path/'meta/feature_signature.json',sig)
            stack.enter_context(patch.object(q,'ROOT',root))
            stack.enter_context(patch.object(q,'history',return_value=(old,base,refs,smfa,[])))
            stack.enter_context(patch.object(q.ab,'parse_args',return_value=args))
            stack.enter_context(patch.object(q.ab,'preflight',side_effect=lambda _:copy.deepcopy(context)))
            stack.enter_context(patch.object(q.ab,'check_gpu'))
            stack.enter_context(patch.object(q.ab,'source_identity',return_value=[]))
            stack.enter_context(patch.object(q.ab,'bank_members',return_value={'x':{'one'}}))
            stack.enter_context(patch.object(q.mpsa,'validate_clean_cache',return_value=dict(count=3)))
            stack.enter_context(patch.object(q.gmp,'enumerate_members',return_value={'t':[]}))
            stack.enter_context(patch.object(q,'ensure_resources'))
            stack.enter_context(patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)))
            stack.enter_context(patch.object(q,'launch',side_effect=launch))
            stack.enter_context(patch.object(q,'merge',side_effect=merge))
            stack.enter_context(patch.object(q,'compare',return_value=dict(candidate_miou=.86,delta_pp=1.)))
            stack.enter_context(patch.dict(os.environ,{},clear=False))
            cli=['--version','queue_test','--mpsa-previous',str(prev),'--clean-proposals-dir',str(clean)]
            q.main(cli)
            output=root/'experiments/queue_test'
            self.assertEqual(q.ab.read(output/'status.json')['completed'],18)
            self.assertEqual(q.ab.read(output/'status.json')['status'],'complete')
            self.assertEqual(sum(s=='train' for s,g,n in launched),17)
            self.assertTrue((output/'train_C_42/attempt_001/failed.json').exists())
            self.assertTrue((output/'train_C_42/attempt_002/queue_worker.json').exists())
            for g in 'AB': self.assertNotIn(('train',g,42),launched)
            count=len(launched); q.main(cli+['--resume']); self.assertEqual(count,len(launched))
            with self.assertRaises(q.GlobalFailure): q.main(cli)


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable locally')
class PoolTests(unittest.TestCase):
    def test_uniform_gmp_equation_and_single_token(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(42); f=torch.randn(8,5)
        value,info=q.gmp.solve(f,torch.ones(8),.1)
        s=f.T@f/8; mu=f.mean(0); expected=F.normalize(torch.linalg.solve(s+.1*s.trace()*torch.eye(5),mu),dim=0)
        torch.testing.assert_close(value,expected,atol=2e-6,rtol=2e-6)
        one,_=q.gmp.solve(f[:1],torch.ones(1),.1)
        torch.testing.assert_close(one,F.normalize(f[0],dim=0),atol=2e-6,rtol=2e-6)
        self.assertLess(info['relative_residual'],1e-4)

    def test_gate_formula_lower_median_and_floor(self):
        import torch
        import torch.nn.functional as F
        f=torch.tensor([[2.,0.],[1.,1.],[-1.,2.],[.2,.3]])
        sim=F.normalize(f,dim=1)@F.normalize(f.mean(0),dim=0)
        w=.5+.5*torch.sigmoid((sim-sim.median())/.1)
        result,info=q.gate_pool(f)
        torch.testing.assert_close(result,F.normalize((f*w[:,None]).sum(0)/w.sum(),dim=0))
        self.assertGreaterEqual(info['gate_min'],.5); self.assertLessEqual(info['gate_max'],1)
        zero,info=q.gate_pool(torch.tensor([[1.,0.],[-1.,0.]])); self.assertEqual(float(zero.norm()),0); self.assertTrue(info['zero_proxy'])
        self.assertEqual(float(q.gate_pool(torch.zeros(0,2))[0].norm()),0)
        with self.assertRaises(ValueError): q.gate_pool(torch.tensor([[float('nan'),1.]]))

    def test_shared_two_forwards_and_single_batch(self):
        import torch
        class Model:
            resolution=1008; context_subtract=0; use_bf16=False; embed_dim=3; calls=0
            def _preprocess_image(self,rgb): return torch.zeros(1,3,1008,1008),None
            def _forward_tokens(self,inp):
                self.calls+=1
                return torch.stack([torch.linspace(1,2,5184),torch.ones(5184),torch.linspace(2,1,5184)],dim=1)[None],(72,72)
        model=Model(); rgb=np.zeros((50,80,3),np.uint8); masks=[np.ones((50,80),bool),np.zeros((50,80),bool)]
        masks[1][4,5]=True
        result=q.extract(model,rgb,masks)
        self.assertEqual(model.calls,2)
        for i,mask in enumerate(masks):
            one=q.extract(model,rgb,[mask])
            for group in q.GROUPS: np.testing.assert_allclose(one[group][0],result[group][i],atol=2e-6,rtol=2e-6)
        self.assertEqual(q.extract(model,rgb,[])['A'].shape,(0,3))


if __name__=='__main__': unittest.main()
