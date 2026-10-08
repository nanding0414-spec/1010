"""Disposable experiment tests; CUDA/checkpoint accuracy is not simulated here."""
from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tools import experimental_pmfa_ab as ab

try:
    import torch
except ImportError:
    torch = None


class InfrastructureTests(unittest.TestCase):
    def test_parallel_builds_and_shared_training_split(self):
        barrier=threading.Barrier(2);trained=threading.Event();pmfa_training=threading.Event()
        calls=[];results={}
        def execute(kind,mode,_cancel=None,**extras):
            calls.append((kind,mode,extras))
            if kind=='build':
                barrier.wait(timeout=5)
                return {'raw':mode+'_raw'}
            if kind=='train':
                if mode=='smfa': trained.set()
                else:
                    self.assertTrue(trained.is_set())
                    self.assertEqual(extras['split_file'],str(Path('smfa_weights')/'training_report.json'))
                    pmfa_training.set()
                return {'weights':mode+'_weights','bank':mode+'_bank'}
            if mode=='smfa': self.assertTrue(pmfa_training.wait(5))
            return {'metrics':mode+'_metrics'}
        ab.parallel_arms(execute,{'pseudo':'shared_labels','proposals':'shared_proposals'},'cleanup',results,lambda:None)
        self.assertEqual(set(results),{'smfa','pmfa'})
        for kind,mode,extras in calls:
            if kind=='build': self.assertEqual(extras['pseudo'],'shared_labels')
            if kind=='eval': self.assertEqual(extras['proposals'],'shared_proposals')

    def test_parallel_failure_cancels_waiting_peer(self):
        barrier=threading.Barrier(2);results={}
        def execute(kind,mode,**extras):
            self.assertEqual(kind,'build')
            barrier.wait(timeout=5)
            if mode=='smfa': raise RuntimeError('baseline failed')
            return {'raw':'pmfa_raw'}
        with self.assertRaisesRegex(RuntimeError,'baseline failed'):
            ab.parallel_arms(execute,{'pseudo':'labels'},'cleanup',results,lambda:None)
        self.assertEqual(results,{})

    def test_parallel_gpu_arguments(self):
        self.assertEqual(ab.parse_args(['--version','dual','--parallel-gpus','0','1']).parallel_gpus,[0,1])
        with self.assertRaises(SystemExit):
            ab.parse_args(['--version','dual','--parallel-gpus','1','1'])

    def test_parameter_validation(self):
        for settings in ({'temperature':0},{'temperature':float('nan')},{'alpha':2},
                         {'steps':-1},{'steps':1.2},{'min_support':0},{'core_threshold':-1},{'rel_margin':-1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ab.PMFA(**settings)

    def test_defaults_and_environment_isolation(self):
        args=ab.parse_args(['--version','test_v1'])
        self.assertEqual(args.gpu,1)
        self.assertEqual(args.steps,2)
        with patch.dict('os.environ',{'OVERRIDE_SAM3_RESOLUTION':'2016','OVERRIDE_REGION_POOLING':'old'}):
            env=ab.base_env(1)
        self.assertEqual(env['OVERRIDE_SAM3_RESOLUTION'],'1008')
        self.assertNotIn('OVERRIDE_REGION_POOLING',env)
        self.assertEqual(env['OVERRIDE_SAM3_USE_BF16'],'false')

    def test_ids_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=root/'ids.txt';ids.write_text('x\nx\n')
            with self.assertRaises(ValueError): ab.get_ids(ids)
            Image.fromarray(np.zeros((4,5),dtype=np.uint8)).save(root/'x.png')
            self.assertEqual(ab.validate_cache(root,'pseudo',['x'],{'x':[5,4]}),[])
            self.assertTrue(ab.validate_cache(root,'pseudo',['x','missing'],{'x':[5,4],'missing':[5,4]}))
            np.savez(root/'x.npz',masks=np.zeros((0,1,1),dtype=np.uint8),scores=np.zeros(0))
            self.assertEqual(ab.validate_cache(root,'proposals',['x'],{'x':[5,4]}),[])
            np.savez(root/'x.npz',masks=np.zeros((1,4,5)),scores=np.array([float('nan')]))
            self.assertTrue(ab.validate_cache(root,'proposals',['x'],{'x':[5,4]}))

    def test_gpu_occupied_stops(self):
        with patch.object(ab.subprocess,'check_output',return_value='Quadro RTX 6000, 24576, 1200, 0'):
            with self.assertRaises(RuntimeError): ab.check_gpu(1)
        with patch.object(ab.subprocess,'check_output',return_value='Quadro RTX 6000, 24576, 20, 0'):
            self.assertEqual(ab.check_gpu(1)['physical_gpu'],1)

    def test_gpu_transient_utilization_settles(self):
        with patch.object(ab.subprocess,'check_output',side_effect=[
                'Quadro RTX 6000, 24576, 16, 26','Quadro RTX 6000, 24576, 16, 0']), \
             patch.object(ab.time,'sleep') as sleep:
            self.assertEqual(ab.check_gpu(1)['utilization'],0)
            sleep.assert_called_once_with(2)

    def test_gpu_persistent_utilization_stops(self):
        with patch.object(ab.subprocess,'check_output',return_value='Quadro RTX 6000, 24576, 16, 26'), \
             patch.object(ab.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'occupied'): ab.check_gpu(1)
            self.assertEqual(sleep.call_count,15)

    def test_fp32_wrapper_applies_before_model_is_returned(self):
        events=[]
        builder=types.ModuleType('sam3.model_builder')
        builder.__file__=str(ab.ROOT/'sam3-main/sam3/model_builder.py')
        class Model:
            def float(self): events.append('float');return self
        builder.build_sam3_image_model=lambda *a,**k: (events.append('build') or Model())
        sam=types.ModuleType('sam3');sam.model_builder=builder
        precision=types.ModuleType('src.sam3_precision')
        precision.use_standard_vit_mlp=lambda model: (events.append('standard_mlp') or 4)
        fake_torch=types.ModuleType('torch')
        fake_torch.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                                 cudnn=types.SimpleNamespace(allow_tf32=True))
        with patch.dict('sys.modules',{'torch':fake_torch,'sam3':sam,'sam3.model_builder':builder,'src.sam3_precision':precision}):
            ab.install_precision()
            builder.build_sam3_image_model()
        self.assertEqual(events,['build','standard_mlp','float'])
        self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)

    def test_compare_rejects_incomplete_val(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('smfa','pmfa'):
                ab.write(root/f'{name}.json',dict(evaluated_ids=['one'],evaluated_count=1,eval_limit=0,mIoU=.5))
            results={m:{'metrics':str(root/f'{m}.json')} for m in ('smfa','pmfa')}
            with self.assertRaises(ValueError): ab.compare(results,{'val_ids':['one','two']})

    def test_compare_real_report_math_and_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results={}
            for mode,score in [('smfa',.5),('pmfa',.55)]:
                sig=ab.signature({},mode,ab.PMFA())
                ab.write(root/mode/'meta/feature_signature.json',sig)
                ab.write(root/mode/'training_report.json',dict(train_images=['a'],validation_images=['b']))
                metric=dict(evaluated_ids=['val'],evaluated_count=1,eval_limit=0,mIoU=score,
                            cleanup={},proposals_dir='same',proposal_count_before=1,proposal_count_after=1,
                            inference_settings={'INF_FEATURE_BANK_ROOT':mode,'INF_NMS_IOU':.4},
                            feature_signature=sig,per_class_iou={'background':score},confusion_matrix=[[1,0],[0,1]])
                ab.write(root/f'{mode}.json',metric)
                results[mode]=dict(metrics=str(root/f'{mode}.json'),bank=str(root/mode),weights=str(root/mode),raw=mode)
            with patch.object(ab,'bank_members',side_effect=[{'background':{'a','b'}},{'background':{'a','c'}}]):
                result=ab.compare(results,{'val_ids':['val']})
            self.assertAlmostEqual(result['delta_pp'],5)
            self.assertEqual(result['bank_membership']['background']['smfa_only'],['b'])
            value=ab.read(root/'pmfa.json');value['inference_settings']['INF_NMS_IOU']=.8
            ab.write(root/'pmfa.json',value)
            with self.assertRaises(ValueError): ab.compare(results,{'val_ids':['val']})

    def test_failed_attempt_and_resume_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); stages=ab.Stages(root,False)
            def fail(attempt):
                (attempt/'partial').write_text('keep')
                raise RuntimeError('fail')
            with self.assertRaises(RuntimeError): stages.run('build',fail)
            self.assertTrue((root/'build/attempt_001/partial').exists())
            def good(attempt):
                f=attempt/'data';f.write_text('original')
                return {'file':str(f)},[attempt]
            result=stages.run('build',good)
            self.assertIn('attempt_002',result['file'])
            with self.assertRaises(FileExistsError): stages.run('build',good)
            stages.resume=True
            self.assertEqual(stages.run('build',lambda _:self.fail('must not run')),result)
            Path(result['file']).write_text('modified')
            with self.assertRaises(ValueError): stages.run('build',good)

    def test_signature_rejects_wrong_method_parameters_and_weights(self):
        from src import feature_signature as fs
        base=dict(format_version=2,backbone='sam3',feature_source='trunk_fpn',feature_level=-1,
                  resolution=1008,token_norm='layernorm',context_subtract=0,embed_dim=4,
                  feature_grid=[72,72],use_bf16=False,model_name='sam3')
        smfa=ab.signature(base,'smfa',ab.PMFA());pmfa=ab.signature(base,'pmfa',ab.PMFA())
        with tempfile.TemporaryDirectory() as directory:
            fs.save_feature_signature(directory,pmfa)
            # Unmodified production validator refuses experimental format.
            with self.assertRaises(RuntimeError): fs.validate_feature_signature(directory,base)
            with patch.object(fs,'REQUIRED_SIGNATURE_KEYS',fs.REQUIRED_SIGNATURE_KEYS|{'experimental_pooling'}):
                fs.validate_feature_signature(directory,pmfa)
                for invalid in (smfa,ab.signature(base,'pmfa',ab.PMFA(alpha=.2)),dict(pmfa,adapter={'sha256':'wrong'})):
                    with self.assertRaises(RuntimeError): fs.validate_feature_signature(directory,invalid)

    def test_full_sequence_shared_inputs_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=ab.parse_args(['--version','ab_test','--dataset-root',directory,'--image-dir',directory,'--gt-dir',directory])
            context=dict(train_ids=['train'],val_ids=['val'],images={},sizes={'train':[5,4],'val':[5,4]},
                         selected={'pseudo':None,'proposals':None},inputs=[],code=[])
            calls=[];fail_new=[True]
            def fake(job,attempt,env,log):
                calls.append((job['mode'],job['stage']))
                ab.write(attempt/'worker_report.json',{'status':'complete','seconds':1})
                log.parent.mkdir(parents=True,exist_ok=True);log.write_text('fake')
                if job['stage']=='pseudo':
                    dest=Path(job['destination']);dest.mkdir(parents=True)
                    Image.fromarray(np.zeros((4,5),dtype=np.uint8)).save(dest/'train.png')
                elif job['stage']=='proposals':
                    dest=Path(job['destination']);dest.mkdir(parents=True)
                    np.savez(dest/'val.npz',masks=np.zeros((0,4,5),dtype=np.uint8),scores=np.zeros(0))
                elif job['stage']=='build':
                    self.assertTrue(Path(job['pseudo']).is_dir())
                    Path(env['OVERRIDE_OUTPUT_ROOT']).mkdir(parents=True)
                elif job['stage']=='train':
                    ab.write(Path(job['weights'])/'training_report.json',{'train_images':['train'],'validation_images':['held']})
                    Path(job['bank']).mkdir(parents=True)
                    if job['mode']=='pmfa':
                        self.assertIn('smfa',job['split_file'])
                elif job['stage']=='eval':
                    self.assertTrue(Path(job['proposals']).is_dir())
                    if job['mode']=='pmfa' and fail_new[0]: raise RuntimeError('new failed')
                    ab.write(job['metrics'],{'mIoU':.5})
            fake_result=dict(per_class_iou={},smfa_miou=.5,pmfa_miou=.6,delta_pp=10,outcome='提高')
            with patch.object(ab,'ROOT',root),patch.object(ab,'launch',side_effect=fake),\
                 patch.object(ab,'source_identity',return_value=[]),patch.object(ab,'compare',return_value=fake_result):
                with self.assertRaises(RuntimeError): ab.run_pipeline(args,context)
                expected=[('smfa','probe'),('smfa','pseudo'),('smfa','proposals')]+[(m,s) for m in ('smfa','pmfa') for s in ('build','train','eval')]
                self.assertEqual(calls,expected)
                out=root/'experiments/ab_test'
                self.assertTrue((out/'smfa_eval/complete.json').exists())
                self.assertFalse((out/'comparison.json').exists())
                self.assertEqual(ab.read(out/'status.json')['status'],'incomplete')
                self.assertFalse((out/'launcher.lock').exists())
                fail_new[0]=False;args.resume=True;calls.clear()
                ab.run_pipeline(args,context)
                self.assertEqual(calls,[('pmfa','eval')])
                self.assertTrue((out/'pmfa_eval/attempt_002').exists())
                self.assertEqual(ab.read(out/'status.json')['status'],'complete')
                args.resume=False
                with self.assertRaises(ValueError): ab.run_pipeline(args,context)

    def test_baseline_failure_never_runs_pmfa(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=ab.parse_args(['--version','failed_baseline'])
            context=dict(train_ids=[],val_ids=[],images={},sizes={},selected={'pseudo':'external','proposals':'external'},inputs=[],code=[])
            calls=[]
            def fake(job,attempt,env,log):
                calls.append((job['mode'],job['stage']))
                if job['stage']=='build': raise RuntimeError('baseline failed')
                ab.write(attempt/'worker_report.json',dict(status='complete'))
                log.parent.mkdir(parents=True,exist_ok=True);log.write_text('fake')
            with patch.object(ab,'ROOT',root),patch.object(ab,'launch',side_effect=fake):
                with self.assertRaises(RuntimeError): ab.run_pipeline(args,context)
            self.assertEqual(calls,[('smfa','probe'),('smfa','build')])


@unittest.skipUnless(torch is not None,'requires torch; not available in lightweight local runtime')
class PoolingTests(unittest.TestCase):
    def original(self,t,w):
        import torch.nn.functional as F
        if w.sum()<1e-6: return torch.zeros(t.shape[1],device=t.device)
        return F.normalize((t*w[:,None]).sum(0)/(w.sum()+1e-8),dim=0)

    def test_alpha_zero_exact_and_small_fallback(self):
        torch.manual_seed(42);t=torch.randn(10,8);w=torch.rand(10)
        self.assertTrue(torch.equal(ab.pool(t,w,ab.PMFA(alpha=0),self.original),self.original(t,w)))
        for w in (torch.zeros(10),torch.tensor([1.,0,0,0,0,0,0,0,0,0])):
            self.assertTrue(torch.equal(ab.pool(t,w,ab.PMFA(),self.original),self.original(t,w)))

    def test_no_core_single_core_zero_tokens_and_invalid(self):
        t=torch.eye(5)
        for w in (torch.full((5,),.2),torch.tensor([1.,.1,.1,.1,.1])):
            v=ab.pool(t,w,ab.PMFA(),self.original)
            self.assertTrue(torch.isfinite(v).all());self.assertAlmostEqual(float(v.norm()),1,places=5)
        self.assertTrue(torch.equal(ab.pool(torch.zeros(5,5),torch.ones(5),ab.PMFA(),self.original),torch.zeros(5)))
        for w in (torch.full((5,),float('nan')),torch.full((5,),-1.)):
            with self.assertRaises(ValueError): ab.pool(t,w,ab.PMFA(),self.original)
        with self.assertRaises(ValueError): ab.pool(t,torch.ones(3),ab.PMFA(),self.original)
        t[0,0]=float('inf')
        with self.assertRaises(ValueError): ab.pool(t,torch.ones(5),ab.PMFA(),self.original)

    def test_matches_reference_formula(self):
        import torch.nn.functional as F
        t=torch.tensor([[1.,0],[.9,.1],[0,1],[.8,.2]])
        w=torch.tensor([1.,.8,.3,.7]);p=ab.PMFA(steps=0)
        q=F.normalize(t[0],dim=0)
        a=torch.softmax(F.normalize(t,dim=1)@q/p.temperature+w.log(),dim=0)
        expected=F.normalize((1-p.alpha)*(t*w[:,None]).sum(0)/(w.sum()+1e-8)+p.alpha*(t*a[:,None]).sum(0),dim=0)
        torch.testing.assert_close(ab.pool(t,w,p,self.original),expected)

    def test_coverage_roundoff_and_real_invalid_values(self):
        t=torch.eye(5)
        w=torch.tensor([1+2**-23, .5, .25, -2**-24, 0.])
        saved=w.clone();stats=Counter()
        actual=ab.pool(t,w,ab.PMFA(),self.original,stats)
        torch.testing.assert_close(actual,ab.pool(t,w.clamp(0,1),ab.PMFA(),self.original))
        self.assertTrue(torch.equal(w,saved))
        self.assertEqual(stats['coverage_roundoff_tokens'],2)
        self.assertTrue(torch.equal(ab.pool(t,w,ab.PMFA(alpha=0),self.original),self.original(t,w)))
        for bad in (-.01,1.01,float('nan'),float('inf')):
            broken=w.clone();broken[0]=bad
            for alpha in (0.,.5):
                with self.assertRaises(ValueError):
                    ab.pool(t,broken,ab.PMFA(alpha=alpha),self.original)

    def test_opencv_area_coverage_roundoff(self):
        if importlib.util.find_spec('cv2') is None: self.skipTest('requires cv2')
        import cv2
        coverage=cv2.resize(np.ones((1008,1008),np.float32),(72,72),interpolation=cv2.INTER_AREA)
        w=torch.from_numpy(coverage.reshape(-1));t=torch.ones(len(w),2)
        actual=ab.pool(t,w,ab.PMFA(),self.original)
        self.assertTrue(torch.isfinite(actual).all())

    def test_single_batch_share_forward_and_core(self):
        if not all(importlib.util.find_spec(m) for m in ('cv2','torchvision')): self.skipTest('requires cv2/torchvision')
        from src.feature_extractor import BaseFeatureExtractor as Base
        from src.feature_extractor_sam3 import SAM3FeatureExtractor as Sam
        from src import feature_signature as fs
        class Dummy(Base):
            def __init__(self):
                super().__init__(torch.device('cpu'),(1,1),4,'dummy');self.calls=0
            def _preprocess_image(self,rgb): return None,(0,0,2,2)
            def _forward_tokens(self,inp):
                self.calls+=1
                return torch.eye(4)[None],(2,2)
        with patch.object(Base,'_pool_tokens',Base._pool_tokens),patch.object(Base,'extract_features_batch',Base.extract_features_batch),\
             patch.object(Sam,'feature_signature',Sam.feature_signature),patch.object(fs,'REQUIRED_SIGNATURE_KEYS',set(fs.REQUIRED_SIGNATURE_KEYS)):
            ab.install_pooling('pmfa',ab.PMFA(),Counter());d=Dummy()
            image=np.zeros((2,2,3),dtype=np.uint8);mask=np.ones((2,2),dtype=bool)
            expected=d.extract_features(image,mask).numpy();d.calls=0
            output=d.extract_features_batch(image,[mask]*33)
            self.assertEqual(d.calls,1)
            for row in output: np.testing.assert_allclose(row,expected,atol=1e-6)


if __name__=='__main__': unittest.main()
