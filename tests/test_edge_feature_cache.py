import contextlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from src.mask_edge_repair import binary_close, edge_postprocess_mask, sam3_logits_to_repaired_mask


class EdgeRepairTests(unittest.TestCase):
    def test_probability_threshold_and_raw_logits_have_same_meaning(self):
        probs=np.array([[.1,.3,.4,.45,.9]])
        expected=np.array([[0,0,0,1,1]],np.uint8)
        np.testing.assert_array_equal(sam3_logits_to_repaired_mask(probs,close_iterations=0,dilate_pixels=0),expected)
        logits=np.log(probs/(1-probs))
        np.testing.assert_array_equal(sam3_logits_to_repaired_mask(logits,close_iterations=0,dilate_pixels=0,
                                                                input_is_probability=False),expected)
        aggressive=sam3_logits_to_repaired_mask(probs,.25,close_iterations=0,dilate_pixels=0)
        self.assertEqual(aggressive.sum(),4)

    def test_close_fills_gap_and_dilate_adds_one_pixel_without_cross_class_mix(self):
        mask=np.zeros((11,11),np.uint8);mask[3:8,3:8]=1;mask[5,5]=0
        closed=binary_close(mask)
        self.assertTrue(closed[5,5]);self.assertFalse(closed[2,4])
        repaired=edge_postprocess_mask(mask)
        self.assertEqual(repaired.dtype,np.uint8);self.assertTrue(repaired[2,4]);self.assertFalse(repaired[1,4])
        self.assertEqual(set(np.unique(repaired)),{0,1})
        self.assertEqual(edge_postprocess_mask(np.zeros((5,5),np.uint8)).sum(),0)
        self.assertTrue(edge_postprocess_mask(np.ones((5,5),np.uint8)).all())

    def test_invalid_parameters_raise(self):
        for kwargs in ({'mask_logit_threshold':float('nan')},{'mask_logit_threshold':0},
                       {'close_kernel':2},{'close_iterations':-1},{'dilate_pixels':-1}):
            with self.assertRaises(ValueError):sam3_logits_to_repaired_mask(np.zeros((3,3)),**kwargs)

    def test_known_presence_keeps_low_presence_classes_and_uses_probability_cutoff(self):
        import torch
        from tools.proposal_demo.test_active import fake_sam3,FakeModel
        from tools.proposal_demo.active_utils import PromptClass
        from tools.proposal_demo.active_proposer import SAM3ActiveProposer
        with fake_sam3():
            model=FakeModel()
            classes=[PromptClass(name=n,prompt=n,hypernyms=()) for n in ['cat','dog','bus','bird']]
            original=model._run_segmentation_heads
            def soft_masks(out,ids):
                original(out,ids);out['pred_masks'][out['pred_masks']>0]=torch.logit(torch.tensor(.45))
            model._run_segmentation_heads=soft_masks
            proposer=SAM3ActiveProposer(model,torch.device('cpu'),classes,min_mask_area=1,
                min_active_classes=1,max_active_classes=1,confidence_threshold=.5,
                force_presence_one=True,mask_probability_threshold=.4)
            result=proposer(np.zeros((32,40,3),np.uint8),class_indices=[0,1,2,3])
            self.assertEqual(set(proposer.last_result['class_ids']),{1,2,3,4})
            self.assertEqual(proposer.last_result['report']['preview_forward_calls'],0)
            self.assertEqual(proposer.last_result['report']['preview_presence'],[1.,1.,1.,1.])
            self.assertTrue((result['instances'].scores>.9).all())
            self.assertNotIn(672,model.image_resolutions)
            with self.assertRaises(ValueError):proposer(np.zeros((32,40,3),np.uint8))

    def test_pseudo_channels_use_table_override_and_preserve_overlap_priority(self):
        import torch
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
        generator=SAM3PseudoMaskGenerator.__new__(SAM3PseudoMaskGenerator)
        generator.classes=['cat','table'];generator.dataset_type='voc';generator.use_bf16=False
        generator.device=torch.device('cpu');generator.prompt_backend='active';generator.edge_repair=True
        generator.edge_options=dict(close_kernel=3,close_iterations=1,dilate_pixels=1,diningtable_close_iterations=2)
        masks=np.zeros((2,12,12),bool);masks[0,5,5]=1;masks[1,5:8,6:9]=1
        class Proposer:
            last_result={'class_ids':np.array([1,2]),'report':{}}
            def __call__(self,*_,**__):return {'instances':types.SimpleNamespace(pred_masks=torch.from_numpy(masks))}
        generator.active_proposer=Proposer()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.jpg';Image.new('RGB',(12,12)).save(path)
            with patch('src.mask_edge_repair.edge_postprocess_mask',wraps=edge_postprocess_mask) as repair:
                output=generator.generate_mask(path,[1,2])
            self.assertEqual([call.args[2] for call in repair.call_args_list],[1,2])
            self.assertEqual(output[5,6],1);self.assertEqual(output[7,8],2)
            self.assertTrue(set(np.unique(output))<={0,1,2})


class ImageCacheTests(unittest.TestCase):
    def test_lossless_roundtrip_avoids_encoder_and_binds_signature_source(self):
        import torch
        from src.image_token_cache import ImageTokenCache
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'MODUSEG_MIN_FREE_BYTES':'0'}):
            root=Path(directory);image=root/'a.jpg';image.write_bytes(b'source')
            rgb=np.zeros((3,4,3),np.uint8);tokens=torch.arange(12,dtype=torch.float32).reshape(1,4,3)
            calls=[]
            model=types.SimpleNamespace(device=torch.device('cpu'),_preprocess_image=lambda _: (None,None),
                _forward_tokens=lambda _: (calls.append(1) or tokens,(2,2)))
            signature={'embed_dim':3,'feature_grid':[2,2],'feature_source':'trunk_fpn','aggregation':{'rho':.1}}
            cache=ImageTokenCache(root/'cache',signature);cache.bind(image,rgb)
            live,grid=cache.tokens(model,rgb);self.assertEqual(calls,[1])
            again=ImageTokenCache(root/'cache',dict(signature,aggregation={'rho':.2}));again.bind(image,rgb)
            loaded,_=again.tokens(model,rgb)
            self.assertTrue(torch.equal(live,loaded));self.assertEqual(calls,[1])
            with np.load(root/'cache/a.npz') as values:
                self.assertEqual(values['tokens'].shape,(2,2,3));self.assertEqual(values['tokens'].dtype,np.float32)
            bad=ImageTokenCache(root/'cache',dict(signature,feature_source='fpn'));bad.bind(image,rgb)
            with self.assertRaises(ValueError):bad.tokens(model,rgb)
            image.write_bytes(b'changed');bad=ImageTokenCache(root/'cache',signature);bad.bind(image,rgb)
            with self.assertRaises(ValueError):bad.tokens(model,rgb)

    def test_cache_checksum_and_unbound_source_are_rejected(self):
        import torch
        from src.image_token_cache import ImageTokenCache
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'MODUSEG_MIN_FREE_BYTES':'0'}):
            root=Path(directory);source=root/'a.jpg';source.write_bytes(b'a');rgb=np.zeros((1,1,3),np.uint8)
            model=types.SimpleNamespace(device=torch.device('cpu'),_preprocess_image=lambda _: (None,None),
                _forward_tokens=lambda _: (torch.ones((1,1,2)),(1,1)))
            cache=ImageTokenCache(root/'cache',{'embed_dim':2,'feature_grid':[1,1]})
            with self.assertRaises(RuntimeError):cache.tokens(model,rgb)
            cache.bind(source,rgb);cache.tokens(model,rgb)
            (root/'cache/a.npz').write_bytes(b'changed')
            cache.live=None
            with self.assertRaises(ValueError):cache.tokens(model,rgb)


if __name__=='__main__':unittest.main()
