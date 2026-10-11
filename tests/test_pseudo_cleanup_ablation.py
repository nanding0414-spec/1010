import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.proposal_cleanup import CleanupConfig, clean_mask
from src.pseudo_label_cleanup import clean_semantic_labels
from tools.run_pseudo_cleanup_ablation import clean_images, region_reuse_plan, checked_source
from tools.run_mainline_full import identity, read


class SemanticCleanupTests(unittest.TestCase):
    def test_uses_online_geometry_and_preserves_large_object_edges(self):
        labels=np.zeros((100,100),np.uint8);labels[20:80,20:80]=1
        labels[30:32,30:32]=0;labels[1:3,1:3]=1
        expected,_,_=clean_mask(labels==1,CleanupConfig())
        result,report=clean_semantic_labels(labels)
        np.testing.assert_array_equal(result==1,expected)
        self.assertEqual(report['removed_island_pixels'],4)
        self.assertEqual(report['filled_background_pixels'],4)
        self.assertTrue(result[20:80,20:80].all())
        self.assertEqual(set(np.unique(result)),{0,1})

    def test_holes_cannot_overwrite_another_foreground_class(self):
        labels=np.ones((60,60),np.uint8);labels[30:34,30:34]=2
        result,report=clean_semantic_labels(labels,CleanupConfig(min_mask_area=0,min_island_area=0))
        self.assertTrue((result[30:34,30:34]==2).all())
        self.assertEqual(report['protected_other_class_pixels'],16)
        self.assertEqual(report['filled_background_pixels'],0)

    def test_area_and_hole_fraction_gates(self):
        labels=np.zeros((50,50),np.uint8);labels[2:11,2:11]=1
        result,report=clean_semantic_labels(labels)
        self.assertFalse(result.any());self.assertEqual(report['area_rejected_classes'],1)
        ring=np.zeros((20,20),np.uint8);ring[2:18,2:18]=1;ring[8:12,8:12]=0
        result,_=clean_semantic_labels(ring)
        np.testing.assert_array_equal(result,ring)
        with self.assertRaises(ValueError):clean_semantic_labels(np.full((5,5),255,np.uint8))

    def test_resume_is_bound_to_source_parameters_and_output_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();output=root/'new'
            labels=np.ones((20,20),np.uint8);Image.fromarray(labels).save(source/'a.png')
            original=(source/'a.png').read_bytes()
            m=dict(root=str(root),source_pseudo=str(source),label_cleanup=CleanupConfig().to_dict(),
                cleanup_policy='geometry',code_digest='code',reused_artifacts={str(source/'a.png'):identity(source/'a.png')})
            with patch('tools.run_mainline_full.free_space'):
                clean_images(m,['a'],output);clean_images(m,['a'],output)
                self.assertEqual((source/'a.png').read_bytes(),original)
                changed=dict(m,label_cleanup=dict(m['label_cleanup'],min_island_area=999))
                with self.assertRaises(ValueError):clean_images(changed,['a'],output)
                (output/'a.png').write_bytes(b'changed')
                with self.assertRaises(ValueError):clean_images(m,['a'],output)
                self.assertEqual((output/'a.png').read_bytes(),b'changed')
                (source/'a.png').write_bytes(b'changed input')
                with self.assertRaises(ValueError):checked_source(m,source/'a.png')

    def test_region_reuse_requires_same_class_and_selected_token_support(self):
        old=np.zeros((1008,1008),np.uint8);old[7,7]=1
        off_center=old.copy();off_center[2,2]=1
        other_token=old.copy();other_token[21,21]=1
        before=[dict(cls_id=1,mask=old)]
        after=[dict(cls_id=1,mask=off_center),dict(cls_id=2,mask=off_center),dict(cls_id=1,mask=other_token)]
        self.assertEqual(region_reuse_plan(before,after),[0,None,None])


class ReadOnlyCacheTests(unittest.TestCase):
    def test_missing_read_only_cache_never_runs_encoder_or_creates_folder(self):
        import torch
        from src.image_token_cache import ImageTokenCache
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'a.jpg';source.write_bytes(b'a')
            rgb=np.zeros((2,2,3),np.uint8)
            def forbidden(_):raise AssertionError('Encoder must not execute')
            model=types.SimpleNamespace(device=torch.device('cpu'),_preprocess_image=forbidden,_forward_tokens=forbidden)
            cache=ImageTokenCache(root/'missing',{'embed_dim':3,'feature_grid':[2,2]},read_only=True)
            cache.bind(source,rgb)
            with self.assertRaises(FileNotFoundError):cache.tokens(model,rgb)
            self.assertFalse((root/'missing').exists())

    def test_read_only_cache_reuses_lossless_data_and_rejects_corruption(self):
        import torch
        from src.image_token_cache import ImageTokenCache
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'MODUSEG_MIN_FREE_BYTES':'0'}):
            root=Path(directory);source=root/'a.jpg';source.write_bytes(b'a');rgb=np.zeros((2,2,3),np.uint8)
            tokens=torch.arange(12,dtype=torch.float32).reshape(1,4,3)
            model=types.SimpleNamespace(device=torch.device('cpu'),_preprocess_image=lambda _: (None,None),
                _forward_tokens=lambda _: (tokens,(2,2)))
            signature={'embed_dim':3,'feature_grid':[2,2]}
            writer=ImageTokenCache(root/'cache',signature);writer.bind(source,rgb);writer.tokens(model,rgb)
            before={p.name:p.read_bytes() for p in (root/'cache').iterdir()}
            readonly=ImageTokenCache(root/'cache',signature,read_only=True);readonly.bind(source,rgb)
            loaded,_=readonly.tokens(model,rgb);self.assertTrue(torch.equal(loaded,tokens))
            self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'cache').iterdir()})
            (root/'cache/a.npz').write_bytes(b'bad');readonly.live=None
            with self.assertRaises(ValueError):readonly.tokens(model,rgb)

    def test_cached_extractor_refuses_adapter_and_wrong_aggregation(self):
        from src.cached_gmp_extractor import CachedGMPExtractor
        from src.hard_center_gmp import describe
        signature=dict(backbone='sam3',feature_source='trunk_fpn',feature_level=-1,resolution=1008,
            token_norm='layernorm',context_subtract=0.0,embed_dim=1280,feature_grid=[72,72],use_bf16=False,
            precision_policy='standard_vit_mlp_fp32_v1',aggregation=describe(.1))
        CachedGMPExtractor('unused',signature,'cpu')
        for changed in [dict(signature,aggregation=describe(.2)),dict(signature,adapter={'sha':'old'}),
                        dict(signature,use_bf16=True)]:
            with self.assertRaises(ValueError):CachedGMPExtractor('unused',changed,'cpu')


if __name__=='__main__':unittest.main()
