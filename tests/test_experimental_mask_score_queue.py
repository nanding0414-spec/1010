"""Synthetic protocol tests; CUDA replay and real accuracy remain server checks."""
import ast
import copy
from contextlib import ExitStack, nullcontext
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tools import experimental_mask_score_queue as q
from tests.test_experimental_smfa_mlp_queue import metrics


class ScoreTests(unittest.TestCase):
    def test_pixel_max_class_priority_components_and_background(self):
        masks = np.array([[[1,1,0,0]], [[0,1,1,0]], [[0,1,1,0]]], dtype=bool)
        old = np.array([[1,1,1,0]], dtype=np.uint8)
        regions = [dict(cls_id=1,mask=np.array([[1,1,0,0]],np.uint8)),
                   dict(cls_id=1,mask=np.array([[0,0,1,0]],np.uint8)),
                   dict(cls_id=0,mask=np.array([[0,0,0,1]],np.uint8))]
        values = q.region_scores(old,masks,[.6,.8,.99],[1,1,2],[1,2],regions)
        self.assertAlmostEqual(values[0]['score'], .7, places=6)
        self.assertAlmostEqual(values[1]['score'], .8, places=6)
        self.assertEqual(values[2],dict(score=None,weight=1.))
        with self.assertRaises(q.ScoreReplayFailure):
            q.region_scores(old,masks,[.6,.8,.99],[1,1,2],[2,1],regions)

    def test_zero_score_is_covered_and_empty_proposals_background(self):
        region = dict(cls_id=1,mask=np.ones((1,1),np.uint8))
        self.assertEqual(q.region_scores(np.ones((1,1)),np.ones((1,1,1)),[0],[1],[1],[region])[0]['weight'], .5)
        bg = dict(cls_id=0,mask=np.ones((2,2),np.uint8))
        self.assertEqual(q.region_scores(np.zeros((2,2)),np.zeros((0,2,2)),[],[],[],[bg])[0]['score'],None)

    def test_invalid_scores_and_geometry_rejected(self):
        for score in (float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(q.ScoreReplayFailure):
                q.region_scores(np.ones((1,1)),np.ones((1,1,1)),[score],[1],[1],[])
        with self.assertRaises(q.ScoreReplayFailure):
            q.region_scores(np.ones((1,1)),np.zeros((0,1,1)),[],[],[1],[])

    def test_row_alignment_by_class_id_not_input_order(self):
        rows = [dict(cls='cat',id='a_000000',mask_sha256='h',score=.8,weight=.9),
                dict(cls='background',id='a_000000',mask_sha256='b',score=None,weight=1.)]
        records=[(Path('features/background/features.npz'),np.zeros((1,2)),['a_000000']),
                 (Path('features/cat/features.npz'),np.zeros((1,2)),['a_000000'])]
        np.testing.assert_allclose(q.align_weights(records,rows),[1,.9])
        for bad in [rows[:1],rows+rows[:1],[dict(rows[0],weight=.8),rows[1]],
                    [dict(rows[0],id='different'),rows[1]]]:
            with self.assertRaises(ValueError): q.align_weights(records,bad)

    def test_sidecar_integrity_order_mask_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); expected=[dict(index=0,cls='cat',id='i_000000',mask_sha256='h')]
            payload=dict(identity='identity',image='i',rows=[dict(expected[0],score=.8,weight=.9)])
            q.ab.write(root/'i.json',dict(payload=payload,sha256=q.object_hash(payload)))
            self.assertEqual(q.read_score_image(root,'i','identity',expected),payload['rows'])
            for identity,plan in [('wrong',expected),('identity',[dict(expected[0],mask_sha256='wrong')]),
                                  ('identity',[dict(expected[0],id='wrong')])]:
                with self.assertRaises(q.queue.GlobalFailure): q.read_score_image(root,'i',identity,plan)
            payload['rows'][0]['score']=.2
            q.ab.write(root/'i.json',dict(payload=payload,sha256='old'))
            with self.assertRaises(q.queue.GlobalFailure): q.read_score_image(root,'i','identity',expected)

    def test_generation_metadata_never_guesses_missing_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            common=dict(pseudo=tmp)
            with self.assertRaises(q.ScoreReplayFailure): q.generation_recipe(common)
            q.ab.write(Path(tmp)/'meta/generation_trainaug.json',dict(format_version=2))
            with self.assertRaises(q.ScoreReplayFailure): q.generation_recipe(common)


class ProtocolTests(unittest.TestCase):
    def test_worker_seed_depth_and_alpha_reach_real_queue_wrapper(self):
        fake_cuda=SimpleNamespace(is_available=lambda:False,set_device=lambda *_:None,reset_peak_memory_stats=lambda:None,
            synchronize=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,
            OutOfMemoryError=type('FakeOOM',(RuntimeError,),{}))
        fake_torch=SimpleNamespace(cuda=fake_cuda)
        sig=q.queue.signature({},'combo',q.queue.Recipe('C','a'*64))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            job=dict(stage='train',group='C',raw_signature=sig,checkpoint_sha256='a'*64,
                training_mode='weighted',seed=123,alpha=.2,attempt=tmp,
                score_tag=dict(protocol=q.PROTOCOL,formula=q.FORMULA,cache_identity='a'*64,rows_sha256='b'*64))
            # The real combo wrapper supplies seed=123 before our depth/CE wrapper.
            def base_worker(_):
                q.sys.argv=['trainer','--seed','42','--num-layers','3','--split-file','fixed']
                q.ab.runpy.run_path(str(q.ROOT/'tools/train_feature_adapter.py'),run_name='__main__')
            stack.enter_context(patch.dict(sys.modules,torch=fake_torch))
            stack.enter_context(patch.object(q.os,'chdir'))
            stack.enter_context(patch.object(q.sys,'argv',[]))
            stack.enter_context(patch.object(q.sys,'path',list(sys.path)))
            stack.enter_context(patch.object(q.mlp,'install_adapter'))
            stack.enter_context(patch.object(q,'install_confidence_adapter'))
            trainer=stack.enter_context(patch.object(q,'train_weighted'))
            for name in ('PMFA','signature','EXPERIMENT_FORMAT','install_pooling'):
                stack.enter_context(patch.object(q.ab,name,getattr(q.ab,name)))
            stack.enter_context(patch.object(q.ab.runpy,'run_path',q.ab.runpy.run_path))
            stack.enter_context(patch.object(q.ab,'worker',side_effect=base_worker))
            q.worker(job)
            trainer.assert_called_once_with(job)
            args=q.ab.read(Path(tmp)/'effective_training_argv.json')['argv']
            self.assertEqual(args[args.index('--seed')+1],'123')
            self.assertEqual(args[args.index('--num-layers')+1],'2')
            self.assertEqual(q.ab.read(Path(tmp)/'queue_worker.json')['status'],'complete')

    def test_only_fusion_setting_can_change(self):
        ref=metrics(); ref['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT']=.2
        current=copy.deepcopy(ref); current['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT']=.4
        q.validate_eval(current,ref,ref['evaluated_ids'],3609,.4)
        current['inference_settings']['INF_TOPK_NEIGH']=99
        with self.assertRaises(ValueError): q.validate_eval(current,ref,ref['evaluated_ids'],3609,.4)
        self.assertEqual(ref['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT'],.2)

    def test_group_rho_checkpoint_and_original_training_rejected(self):
        sig=q.queue.signature({},'combo',q.queue.Recipe('C','a'*64))
        job=dict(stage='eval',group='C',raw_signature=sig,checkpoint_sha256='a'*64,
                 training_mode='original',seed=42,alpha=.2)
        q.validate_job(job)
        for updates in [dict(stage='train'),dict(group='D'),dict(alpha=.3),dict(seed=0),dict(checkpoint_sha256='b'*64)]:
            with self.assertRaises(ValueError): q.validate_job(dict(job,**updates))

    def test_partial_means_and_prespecified_interaction(self):
        reports={q.cell(mode,seed,a):dict(mIoU=.85+.01*(mode=='weighted')+.001*a)
                 for mode in ('original','weighted') for seed in q.SEEDS for a in q.ALPHAS}
        out=q.summary(reports)
        self.assertAlmostEqual(out['contrasts']['training']['mean_pp'],1.)
        self.assertAlmostEqual(out['contrasts']['interaction_a4']['mean_pp'],0.)
        del reports[q.cell('weighted',2026,.2)]
        out=q.summary(reports)
        self.assertIsNone(out['groups']['weighted_a2']['mean_miou'])
        self.assertIsNone(out['contrasts']['training']['mean_pp'])

    def test_per_image_replay_requires_ids_and_each_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.npz',Path(tmp)/'b.npz'
            x=np.stack([np.eye(21,dtype=int),np.zeros((21,21),int)])
            np.savez(a,ids=['a','b'],confusions=x)
            np.savez(b,ids=['a','b'],confusions=x[::-1])
            with self.assertRaises(q.queue.GlobalFailure): q.require_image_replay(a,b)
            np.savez(b,ids=['a','b'],confusions=x)
            q.require_image_replay(a,b)

    def test_v1_training_core_is_preserved(self):
        old=ast.parse((q.ROOT/'tools/train_feature_adapter.py').read_text(encoding='utf-8'))
        new=ast.parse(Path(q.__file__).read_text(encoding='utf-8'))
        def named_calls(tree,names):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Name) and n.func.id in names]
        old=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        new=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='train_weighted')
        names={'balanced_batch','supervised_contrastive_loss','retrieval_metrics','project_features','split_from_report'}
        self.assertEqual(named_calls(old,names),named_calls(new,names))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable locally; run on server')
class TensorTests(unittest.TestCase):
    def test_all_ones_loss_gradient_and_short_training_exact(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(42)
        a=torch.nn.Linear(4,3); b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.001); ob=torch.optim.AdamW(b.parameters(),lr=.001)
        x=torch.randn(8,4); y=torch.arange(8)%3
        for _ in range(5):
            oa.zero_grad();ob.zero_grad()
            la=F.cross_entropy(a(x),y);lb=q.weighted_ce(b(x),y,torch.ones(8))
            self.assertTrue(torch.equal(la,lb));la.backward();lb.backward()
            for pa,pb in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(pa.grad,pb.grad))
            oa.step();ob.step()
        for pa,pb in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(pa,pb))
        logits=b(x); w=torch.linspace(.5,1,8)
        self.assertTrue(torch.equal(q.weighted_ce(logits,y,w),(F.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum()))

    def test_weighted_checkpoint_provenance_and_source(self):
        import torch
        import src.feature_adapter as fa
        originals=(fa.FeatureAdapter,fa.load_adapter,fa.adapted_signature)
        tag=dict(protocol=q.PROTOCOL,formula=q.FORMULA,cache_identity='a'*64)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                q.mlp.install_adapter('d2');q.install_confidence_adapter(tag)
                model=fa.FeatureAdapter(1280,512,256,2)
                sig=dict(backbone='sam3',embed_dim=1280)
                path=Path(tmp)/'model.pt'
                torch.save(dict(format_version=1,architecture=model.architecture,state_dict=model.state_dict(),source_signature=sig),path)
                loaded,_=fa.load_adapter(path,'cpu',sig)
                self.assertEqual(loaded.architecture['score_training'],tag)
                with self.assertRaises(ValueError): fa.load_adapter(path,'cpu',dict(sig,embed_dim=256))
                payload=torch.load(path,weights_only=True);payload['architecture']['score_training']['cache_identity']='b'*64
                torch.save(payload,path)
                with self.assertRaises(ValueError): fa.load_adapter(path,'cpu',sig)
            finally:
                fa.FeatureAdapter,fa.load_adapter,fa.adapted_signature=originals


class SimulationTests(unittest.TestCase):
    def simulate(self,root,resume=False,failure=None,metadata_error=False):
        source=root/'source';previous=root/'previous'
        q.ab.write(previous/'manifest.json',dict(combo_previous=str(source)))
        split=dict(train_images=['train'],validation_images=['heldout'])
        member=dict(index=0,cls='cat',id='train_000000',mask_sha256='h')
        plan={'train':[member],'heldout':[]}
        q.ab.write(source/'members.json',plan);q.ab.write(source/'membership/complete.json',{})
        raw=root/'raw';raw.mkdir(exist_ok=True)
        sig=dict(format_version=3006)
        refs={}
        for seed in q.SEEDS:
            w,b=root/f'w{seed}',root/f'b{seed}'
            q.ab.write(w/'training_report.json',dict(split,args=dict(seed=seed)));q.ab.write(b/'meta/feature_signature.json',sig)
            met=metrics(sig);met['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT']=.2
            metric=root/f'm{seed}.json';q.ab.write(metric,met)
            per=root/f'p{seed}.npz'; counts=np.zeros((1449,21,21),np.int64);counts[0]=np.eye(21,dtype=int)
            if not per.exists(): np.savez(per,ids=met['evaluated_ids'],confusions=counts)
            refs[seed]=dict(weights=str(w),bank=str(b),metrics=str(metric),per_image=str(per),raw=str(raw))
        common=dict(pseudo=str(root/'pseudo'),proposals=str(root/'proposals'),train_ids=['train','heldout'],
                    val_ids=met['evaluated_ids'],cache_identity={'train':{},'heldout':{}},checkpoint_sha256='a'*64,
                    images={},cache='fixed-cache',checkpoint='fixed-checkpoint')
        calls=[]; real_load=np.load
        def load(path,*args,**kwargs):
            if Path(path).parent==root/'proposals': return nullcontext(dict(scores=np.array([.8])))
            return real_load(path,*args,**kwargs)
        def launch(job,attempt,env,log,cancel,deadline,respect_deadline=True):
            calls.append((job['stage'],job['training_mode'],job['seed'],job['alpha']))
            if failure=='replay' and job['stage']=='eval': raise q.queue.TaskFailure('replay failed')
            if failure=='scores' and job['stage']=='scores': raise q.queue.TaskFailure('PNG mismatch')
            if failure=='train' and job['stage']=='train' and job['seed']==123: raise q.queue.TaskFailure('local train failure')
            q.ab.write(attempt/'queue_worker.json',dict(status='complete',seconds=.1))
            log.parent.mkdir(parents=True,exist_ok=True);log.write_text('simulation',encoding='utf-8')
            if job['stage']=='scores':
                for stem in job['score_ids']:
                    if (Path(job['score_root'])/f'{stem}.json').exists(): continue
                    payload=dict(identity=job['score_identity_hash'],image=stem,
                                 rows=[dict(r,score=.8,weight=.9) for r in job['plan'][stem]])
                    q.ab.write(Path(job['score_root'])/f'{stem}.json',dict(payload=payload,sha256=q.object_hash(payload)))
            elif job['stage']=='train':
                q.ab.write(Path(job['weights'])/'training_report.json',dict(split,score_training=job['score_tag'],args=dict(seed=job['seed'])))
                q.ab.write(Path(job['bank'])/'meta/feature_signature.json',dict(adapter=dict(type=q.PROTOCOL,
                    architecture=dict(score_training=job['score_tag']))))
            else:
                value=metrics(q.ab.read(Path(job['bank'])/'meta/feature_signature.json'))
                value['inference_settings']['INF_PROPOSAL_SCORE_WEIGHT']=job['alpha']
                q.ab.write(Path(job['metrics']),value)
                np.savez(attempt/'per_image_confusion.npz',ids=value['evaluated_ids'],confusions=counts)
        recipe_effect=q.ScoreReplayFailure('missing metadata') if metadata_error else None
        with ExitStack() as stack:
            for target,name,kwargs in [
                (q,'ROOT',dict(new=root)),(q.os,'chdir',{}),
                (q,'checked_history',dict(return_value=(common,refs,split,[],{'count':3609},str(raw),sig))),
                (q,'generation_recipe',dict(return_value=({'recipe':1},[]),side_effect=recipe_effect)),
                (q.queue,'verified_stage',dict(return_value={})),(q.ab,'check_gpu',{}),
                (q.ab,'source_identity',dict(return_value=[])),
                (q.ab,'bank_members',dict(return_value={'cat':{'train_000000'}})),
                (q.queue.shutil,'disk_usage',dict(return_value=SimpleNamespace(free=30*2**30))),
                (q.queue,'launch',dict(side_effect=launch)),(np,'load',dict(side_effect=load))]:
                stack.enter_context(patch.object(target,name,**kwargs))
            args=['--version','score_test','--previous',str(previous)] + (['--resume'] if resume else [])
            if failure=='replay':
                with self.assertRaises(q.queue.GlobalFailure): q.main(args)
            else: q.main(args)
        return calls,q.ab.read(root/'experiments/score_test/summary.json')

    def test_full_grid_history_gates_scores_and_resume_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls,report=self.simulate(root)
            self.assertEqual(report['status'],'complete');self.assertEqual(report['completed'],24)
            self.assertEqual(sum(c[0]=='train' for c in calls),3)
            self.assertTrue(all(c[0]=='eval' for c in calls[:12]))
            self.assertTrue(all(c[3]==.2 for c in calls[:3]))
            calls,_=self.simulate(root,resume=True);self.assertEqual(calls,[])
            with self.assertRaises(ValueError):self.simulate(root)

    def test_replay_failure_never_collects_or_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,report=self.simulate(Path(tmp),failure='replay')
            self.assertEqual(report['status'],'incomplete')
            self.assertTrue(all(c[0]=='eval' for c in calls))

    def test_score_failure_preserves_all_inference_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,report=self.simulate(Path(tmp),failure='scores')
            self.assertEqual(report['completed'],12)
            self.assertEqual(report['score_status'],'blocked_replay')
            self.assertFalse(any(c[0]=='train' for c in calls))

    def test_metadata_missing_does_not_block_independent_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,report=self.simulate(Path(tmp),metadata_error=True)
            self.assertEqual(report['completed'],12)
            self.assertEqual(report['score_status'],'blocked_metadata')

    def test_local_failure_continues_and_resume_only_failed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls,report=self.simulate(root,failure='train')
            self.assertEqual(report['completed'],20)
            calls,report=self.simulate(root,resume=True)
            self.assertEqual(report['completed'],24)
            self.assertEqual(sum(c[0]=='train' for c in calls),1)
            self.assertTrue(all(c[2]==123 for c in calls))

    def test_scheduler_disk_deadline_and_per_gpu_exclusion(self):
        cancel=threading.Event();active=set();lock=threading.Lock();done=[]
        def action(item,gpu):
            with lock:
                self.assertNotIn(gpu,active);active.add(gpu)
            time.sleep(.002)
            with lock:active.remove(gpu);done.append(item)
        with patch.object(q.queue.shutil,'disk_usage',return_value=SimpleNamespace(free=30*2**30)):
            q.queue.run_round(list(range(6)),[0,1],action,cancel,time.time()+30,lambda *_:None)
            self.assertEqual(sorted(done),list(range(6)))
            with self.assertRaises(q.queue.Deferred):
                q.queue.run_round([1],[0,1],action,cancel,time.time()-1,lambda *_:None)
        with patch.object(q.queue.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(q.queue.Deferred):q.queue.ensure_resources(time.time()+30)

    def test_launcher_cancellation_only_terminates_owned_worker(self):
        from unittest.mock import Mock
        cancel=threading.Event()
        process=Mock(pid=43210,returncode=None)
        process.poll.return_value=None
        def spawn(*args,**kwargs):
            cancel.set();return process
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp)
            stack.enter_context(patch.object(q.queue,'ensure_resources'))
            stack.enter_context(patch.object(q.queue.subprocess,'check_output',return_value='16, 0'))
            stack.enter_context(patch.object(q.queue.subprocess,'Popen',side_effect=spawn))
            if q.os.name=='posix':
                kill=stack.enter_context(patch.object(q.queue.os,'killpg'))
            with self.assertRaises(q.queue.GlobalFailure):
                q.queue.launch(dict(stage='eval'),root,{'CUDA_VISIBLE_DEVICES':'1'},root/'worker.log',cancel,time.time()+30)
            if q.os.name=='posix':kill.assert_called_once_with(43210,q.signal.SIGTERM)
            else:process.terminate.assert_called_once()
            process.wait.assert_called_once()


if __name__=='__main__':
    unittest.main()
