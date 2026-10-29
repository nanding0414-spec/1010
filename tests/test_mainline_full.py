import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tools.run_mainline_full import (coverage, identity, checked_checkpoint, write,
    save_npz, merge_metrics, merge_bank, environment)


class FullMainlineTests(unittest.TestCase):
    def test_strict_proposal_resume_does_not_overwrite_changed_cache(self):
        from PIL import Image
        from run_generate_proposals_active import main
        from tools.proposal_demo.test_active import fake_sam3
        with tempfile.TemporaryDirectory() as directory, fake_sam3(), \
             patch.dict('os.environ', {'MODUSEG_STRICT_CACHE': '1'}), contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);Image.new('RGB',(40,32)).save(root/'a.jpg');(root/'sam3.pt').touch()
            args=['--image-dir',str(root),'--checkpoint',str(root/'sam3.pt'),'--output-dir',str(root/'raw'),
                  '--device','cpu','--precision','fp32','--min-mask-area','1']
            self.assertEqual(main(args),0);self.assertEqual(main(args),0)
            path=root/'raw/a.npz';path.write_bytes(b'changed original')
            with self.assertRaisesRegex(ValueError,'cache retained'):
                main(args)
            self.assertEqual(path.read_bytes(),b'changed original')

    def test_shards_require_exact_unique_coverage(self):
        coverage([['a', 'c'], ['b', 'd']], ['a', 'b', 'c', 'd'])
        for shards in ([['a'], ['a']], [['a'], []], [['a'], ['extra']]):
            with self.assertRaises(ValueError):
                coverage(shards, ['a', 'b'])

    def test_checkpoint_resume_checks_hash_input_and_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'a.npz'; binding = {'signature': {'rho': .1}, 'image': 'sha'}
            self.assertFalse(checked_checkpoint(path, binding))
            save_npz(path, features=np.ones((2, 3)))
            with self.assertRaises(ValueError):
                checked_checkpoint(path, binding)
            write(path.with_suffix('.json'), dict(binding=binding, artifact=identity(path)))
            self.assertTrue(checked_checkpoint(path, binding))
            with self.assertRaises(ValueError):
                checked_checkpoint(path, dict(binding, signature={'rho': .2}))
            save_npz(path, features=np.zeros((2, 3)))
            with self.assertRaises(ValueError):
                checked_checkpoint(path, binding)

    def test_metrics_sum_counts_and_reject_settings_or_duplicate_ids(self):
        names = ['background']+[f'c{i}' for i in range(20)]
        def report(stem, matrix):
            return dict(evaluated_ids=[stem], confusion_matrix=matrix.tolist(),
                per_class_iou=dict.fromkeys(names, 0), feature_signature={'rho': .1},
                feature_bank='bank', proposals_dir='clean', inference_settings={'K': 25}, cleanup=None,
                proposal_count_before=5, proposal_count_after=2,
                semantic_rescue=dict(parameters={'margin': .1}, images=[{'id': stem}], accepted=1, added_pixels=3))
        a=np.eye(21, dtype=np.int64); b=np.eye(21, dtype=np.int64)*100; a[0,1]=40
        reports=[report('a',a), report('b',b)]
        actual=merge_metrics(reports,['a','b']); total=a+b
        iou=np.diag(total)/(total.sum(0)+total.sum(1)-np.diag(total))
        self.assertAlmostEqual(actual['mIoU'],float(iou.mean()))
        wrong_mean=((np.diag(a)/(a.sum(0)+a.sum(1)-np.diag(a))).mean()+1)/2
        self.assertNotAlmostEqual(actual['mIoU'],float(wrong_mean))
        self.assertEqual(actual['semantic_rescue']['accepted'],2)
        self.assertEqual(actual['semantic_rescue']['added_pixels'],6)
        with self.assertRaises(ValueError):merge_metrics([reports[0],reports[0]],['a','b'])
        reports[1]['inference_settings']={'K': 5}
        with self.assertRaises(ValueError):merge_metrics(reports,['a','b'])

    def test_environment_removes_ambient_adapter_and_uses_physical_gpu(self):
        manifest={'dataset':'/data/VOC','checkpoint':'/data/sam3.pt','seed':42}
        with patch.dict('os.environ',{'OVERRIDE_FEATURE_ADAPTER_CKPT':'old.pt','OVERRIDE_SAM3_POOLING':'smfa'}):
            env=environment(manifest,Path('/data/code'),1)
        self.assertEqual(env['OVERRIDE_FEATURE_ADAPTER_CKPT'],'')
        self.assertEqual(env['OVERRIDE_SAM3_POOLING'],'hard_center_gmp')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')

    def test_merge_filters_once_globally_and_preserves_image_identity(self):
        from configs import config as cfg
        from src.filter_stage import run_filter_stage
        from src.feature_bank_storage import load_feature_matrix_npz
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cache=root/'cache'; target=root/'bank'; cache.mkdir()
            ids=[f'2007_{i:06d}' for i in range(12)]
            for stem in ids:
                save_npz(cache/f'{stem}.npz',features=np.ones((21,3),np.float32),class_ids=np.arange(21),
                    sample_ids=np.array([f'{stem}_{cid:06d}' for cid in range(21)]))
            from src.feature_signature import REQUIRED_SIGNATURE_KEYS
            signature=root/'signature.json';write(signature,dict({k:None for k in REQUIRED_SIGNATURE_KEYS},embed_dim=3))
            def mapping(*_):
                write(target/'meta/class_mapping.json',{'id_to_dirname':dict(enumerate(['background']+cfg.DATASET_CLASSES))})
            with patch('src.build_feature_bank.save_class_mapping',side_effect=mapping), \
                 patch('src.filter_stage.FEATURES_DIR',str(target/'features')), \
                 patch('src.filter_stage.INDEX_DIR',str(target/'index')), \
                 patch('src.filter_stage.META_DIR',str(target/'meta')), \
                 patch('src.filter_stage.run_filter_stage',wraps=run_filter_stage) as filtering:
                merge_bank(cache,list(reversed(ids)),[signature,signature],target)
                filtering.assert_called_once_with()
            background, samples=load_feature_matrix_npz(target/'features/background/background_features.npz')
            foreground, fg_ids=load_feature_matrix_npz(target/'features'/cfg.DATASET_CLASSES[0]/f'{cfg.DATASET_CLASSES[0]}_features.npz')
            self.assertEqual(len(background),12)
            self.assertEqual(len(foreground),11)  # floor(12*.1), not twice floor(6*.1)
            self.assertEqual(samples[0],ids[0]+'_000000')
            self.assertTrue(all(str(s).rsplit('_',1)[0] in ids for s in fg_ids))

    def test_fp32_proposer_patch_precedes_active_constructor(self):
        import torch
        from tools.proposal_demo.test_active import fake_sam3
        from run_sam3_proposal_demo import parse_args,build_proposer
        model=unittest.mock.MagicMock();model.to.return_value=model;model.eval.return_value=model
        def construct(**kwargs):
            correction.assert_called_once_with(model);model.float.assert_called_once();return kwargs
        args=parse_args(['--precision','fp32'],bulk=True)
        with fake_sam3(), patch('sam3.model_builder.build_sam3_image_model',return_value=model), \
             patch('src.sam3_precision.use_standard_vit_mlp',return_value=1) as correction, \
             patch('tools.proposal_demo.active_proposer.SAM3ActiveProposer',side_effect=construct):
            actual=build_proposer(args,torch.device('cpu'))
        self.assertIs(actual['model'],model)


if __name__=='__main__':unittest.main()
