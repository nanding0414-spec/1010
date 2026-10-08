import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from tools import experimental_view_augmentation_queue as exp


class ViewTests(unittest.TestCase):
    def test_aug_list_exact_membership_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'aug.txt'
            inv={'files':{f'x_{v}':{} for v in exp.VIEWS}}
            names=['x',*inv['files']]
            path.write_text('\n'.join(names)+'\n')
            self.assertEqual(exp.verify_aug_list(path,['x'],inv)['count'],5)
            path.write_text('\n'.join(names+['x']))
            with self.assertRaises(ValueError): exp.verify_aug_list(path,['x'],inv)
            path.write_text('\n'.join(names[:-1]))
            with self.assertRaises(ValueError): exp.verify_aug_list(path,['x'],inv)

    def test_inventory_incomplete_complete_and_original_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(20,20)).save(root/'2007_000001.jpg')
            Image.new('RGB',(20,20)).save(root/'2007_000001_L30.png')
            report=exp.inventory(root,['2007_000001'],['2007_000002'])
            self.assertEqual(len(report['missing']),3)
            self.assertEqual(report['ignored_original_copies'],1)
            for view in exp.VIEWS[1:]: Image.new('RGB',(20,20)).save(root/f'2007_000001_{view}.png')
            report=exp.inventory(root,['2007_000001'],['2007_000002'],True)
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['files']),4)
            self.assertEqual(len(report['files']['2007_000001_L30']['sha256']),64)

    def test_inventory_rejects_val_unknown_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'val_L30.png'; Image.new('RGB',(20,20)).save(p)
            with self.assertRaisesRegex(ValueError,'VOC val'): exp.inventory(root,['train'],['val'])
            p.unlink(); p=root/'train_L90.png'; Image.new('RGB',(20,20)).save(p)
            with self.assertRaisesRegex(ValueError,'Unknown'): exp.inventory(root,['train'],['val'])
            p.unlink()
            for ext in ('jpg','png'): Image.new('RGB',(20,20)).save(root/f'train_L30.{ext}')
            with self.assertRaisesRegex(ValueError,'Duplicate'): exp.inventory(root,['train'],['val'])

    def test_pseudo_union_priority_and_disallowed(self):
        masks=np.array([[[1,1],[0,0]],[[0,1],[1,0]],[[0,0],[0,1]]],bool)
        seg=exp.pseudo_map(masks,[2,1,2],[2,1],(2,2))
        np.testing.assert_array_equal(seg,[[2,2],[1,2]])
        np.testing.assert_array_equal(exp.pseudo_map(np.empty((0,2,2),bool),[],[1],(2,2)),np.zeros((2,2)))
        with self.assertRaises(ValueError): exp.pseudo_map(masks,[2,1,3],[2,1],(2,2))

    def test_angle_sampling_weight_and_no_family_repeat(self):
        views=np.tile(exp.VIEWS,20); groups=np.repeat([f'i{x}' for x in range(20)],4)
        rows=np.arange(len(views)); counts={}
        for method in ('view30','view_all','view_weighted'):
            rng=np.random.default_rng(42); selected=[]
            for _ in range(3000):
                batch=exp.choose_aug_rows(rows,groups,views,['i0'],2,rng,method)
                self.assertEqual(len(set(groups[batch])),2)
                self.assertNotIn('i0',groups[batch]); selected.extend(batch)
            counts[method]=np.mean(np.isin(views[selected],['L60','R60']))
        self.assertEqual(counts['view30'],0)
        self.assertAlmostEqual(counts['view_all'],.5,delta=.03)
        self.assertAlmostEqual(counts['view_weighted'],.2,delta=.03)

    def test_sampler_baseline_equivalence_and_foreground_ratio(self):
        y=np.array([0]*4+[1]*4+[1]*16)
        groups=np.array([f'real{i}' for i in range(8)]+[f'aug{i}' for i in range(16)])
        views=np.array(['real']*8+list(exp.VIEWS)*4)
        original=lambda *args: np.arange(8)
        counts=exp.Counter()
        args=(original,y,groups,np.arange(8),np.random.default_rng(42),2,4,{1:np.arange(8,24)},views,0)
        np.testing.assert_array_equal(exp.augmented_batch(*args,'baseline',counts),np.arange(8))
        rows=exp.augmented_batch(*args,'view_weighted',counts)
        self.assertEqual(sum(rows>=8),2)
        self.assertEqual(sum(y[rows]==0),4)
        self.assertEqual(counts['real_foreground'],2)
        self.assertEqual(len(rows),8)

    def test_relocation_is_boundary_limited_and_does_not_mutate(self):
        original={'path':exp.OLD_ROOT+'/experiments/a.json','external':'/home/zystu2025/datasets/VOC2012',
                  'other':exp.OLD_ROOT+'_backup/a','sha256':'abc'}
        before=copy.deepcopy(original); result=exp.relocate(original)
        self.assertEqual(original,before)
        self.assertEqual(result['path'],str(exp.ROOT/'experiments/a.json'))
        self.assertEqual(result['external'],original['external'])
        self.assertEqual(result['other'],original['other'])
        self.assertEqual(result['sha256'],'abc')

    def test_duplicate_records_conflict(self):
        r=dict(path=str(Path('a').resolve()),size=2,sha256='abc')
        self.assertEqual(len(exp.unique_records([r,r])),1)
        with self.assertRaises(ValueError): exp.unique_records([r,dict(r,sha256='bad')])

    def test_summary_requires_three_seeds(self):
        results={'baseline_42':dict(miou=.85),'view30_42':dict(miou=.86)}
        self.assertIsNone(exp.summarize(results)['groups']['view30']['mean'])
        for seed in (123,2026):
            results[f'baseline_{seed}']=dict(miou=.85)
            results[f'view30_{seed}']=dict(miou=.86)
        summary=exp.summarize(results)
        self.assertAlmostEqual(summary['contrasts']['view30-baseline']['mean_pp'],1)

    def test_cache_identity_binds_angle_hash_and_recipe(self):
        job=dict(view_files={'x_L30':dict(source_id='x',view='L30',sha256='a')},tags={'x':[1]},
                 active_classes=[],raw_signature={'checkpoint':'a'},source_sha='b')
        first=exp.cache_identity(job,'x_L30')
        other=copy.deepcopy(job); other['view_files']['x_L30']['sha256']='changed'
        self.assertNotEqual(first,exp.cache_identity(other,'x_L30'))
        other=copy.deepcopy(job); other['raw_signature']['checkpoint']='changed'
        self.assertNotEqual(first,exp.cache_identity(other,'x_L30'))

    def test_stage_resume_hash_and_failed_attempt_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); stages=exp.AliasAwareStages(root,True)
            def failing(attempt):
                (attempt/'partial.txt').write_text('preserve')
                raise RuntimeError('local failure')
            with self.assertRaises(RuntimeError): stages.run('train',failing)
            self.assertTrue((root/'train/attempt_001/partial.txt').exists())
            def success(attempt):
                file=attempt/'result.txt'; file.write_text('ok')
                return dict(attempt=str(attempt)),[file]
            result=stages.run('train',success)
            self.assertTrue(result['attempt'].endswith('attempt_002'))
            self.assertEqual(stages.run('train',lambda _: self.fail('must skip')),result)
            (Path(result['attempt'])/'result.txt').write_text('corrupt')
            with self.assertRaises(ValueError): stages.run('train',success)


class TrainerTests(unittest.TestCase):
    def test_original_split_and_export_records_unchanged(self):
        from src import adapter_training as at
        real=np.ones((4,1280),np.float32)/np.sqrt(1280)
        groups=np.array(['a','b','v','v']); y=np.array([0,1,0,1]); records=[('real',real,['1','2','3','4'])]
        classes=['background','aeroplane']; signature={'test':1}
        job=dict(raw_signature=signature,split={'train_images':['a','b'],'validation_images':['v']},
                 view_files={'a_L30':{'source_id':'a','view':'L30'},'b_R30':{'source_id':'b','view':'R30'}},method='view30')
        saved=(at.load_training_bank,at.split_from_report,at.balanced_batch)
        try:
            with mock.patch.object(at,'load_training_bank',return_value=(real,y,groups,classes,signature,records)), \
                 mock.patch.object(at,'split_from_report',return_value=(np.array([0,1]),np.array([2,3]))), \
                 mock.patch.object(exp,'read_vectors',return_value=(real[:1],np.array([1]),{})):
                exp.install_training(job)
                x,yy,gg,cc,ss,rr=at.load_training_bank('unused','unused')
                train,val=at.split_from_report(gg,yy,'unused')
                self.assertEqual(len(x),6); self.assertIs(rr,records)
                np.testing.assert_array_equal(train,[0,1]); np.testing.assert_array_equal(val,[2,3])
        finally: at.load_training_bank,at.split_from_report,at.balanced_batch=saved

    def test_generated_validation_family_rejected(self):
        from src import adapter_training as at
        real=np.ones((4,1280),np.float32)
        saved=(at.load_training_bank,at.split_from_report,at.balanced_batch)
        job=dict(raw_signature={},split={'train_images':['a','b'],'validation_images':['v']},
                 view_files={'v_L30':{'source_id':'v','view':'L30'}},method='view30')
        try:
            with mock.patch.object(at,'load_training_bank',return_value=(real,np.array([0,1,0,1]),
                np.array(['a','b','v','v']),['background','aeroplane'],{},[])):
                exp.install_training(job)
                with self.assertRaisesRegex(ValueError,'leakage'): at.load_training_bank('unused','unused')
        finally: at.load_training_bank,at.split_from_report,at.balanced_batch=saved

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable: gradient/trajectory test not executed')
    def test_baseline_short_training_trajectory(self):
        import torch
        from src.adapter_training import balanced_batch
        from src.feature_adapter import supervised_contrastive_loss
        y=np.repeat([0,1],4); families=np.array([f'g{i}' for i in range(8)])
        x=torch.tensor(np.random.default_rng(1).normal(size=(8,5)),dtype=torch.float32)
        states=[]
        for hooked in (False,True):
            torch.manual_seed(1); rng=np.random.default_rng(42)
            model=torch.nn.Linear(5,2); opt=torch.optim.SGD(model.parameters(),lr=.01)
            for _ in range(4):
                if hooked:
                    rows=exp.augmented_batch(balanced_batch,y,families,np.arange(8),rng,2,4,{},
                        np.array(['real']*8),0,'baseline',exp.Counter())
                else: rows=balanced_batch(y,families,np.arange(8),rng,2,4)
                z=model(x[rows]); target=torch.from_numpy(y[rows])
                loss=torch.nn.functional.cross_entropy(z,target)+.5*supervised_contrastive_loss(z,target,.1,0)
                opt.zero_grad(); loss.backward(); opt.step()
            states.append(copy.deepcopy(model.state_dict()))
        for key in states[0]: self.assertTrue(torch.equal(states[0][key],states[1][key]))


if __name__=='__main__': unittest.main()
