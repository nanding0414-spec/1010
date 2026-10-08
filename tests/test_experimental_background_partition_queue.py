import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools import experimental_background_partition_queue as experiment


class GeometryTests(unittest.TestCase):
    def test_vocabulary_tuple_list_roundtrip_compares_by_content(self):
        classes=[dict(name='cat',synonyms=('cat','feline'))]
        serialized=json.loads(json.dumps(classes))
        self.assertNotEqual(classes,serialized)
        self.assertEqual(experiment.old.canonical(classes),experiment.old.canonical(serialized))
        serialized[0]['name']='dog'
        self.assertNotEqual(experiment.old.canonical(classes),experiment.old.canonical(serialized))

    def test_old_train_job_gets_extraction_fields_from_verified_matching_reference(self):
        train=dict(cache='cache',checkpoint_sha256='a'*64,train_ids=['image'],raw_signature={'format_version':3006})
        ref=dict(train,plan='members.json',pseudo='pseudo',source={'format_version':2})
        got=experiment.extraction_context(train,ref)
        self.assertEqual(got['plan'],'members.json');self.assertEqual(got['pseudo'],'pseudo')
        self.assertNotIn('plan',train)
        with self.assertRaises(ValueError):experiment.extraction_context(train,dict(ref,cache='other'))
        with self.assertRaises(ValueError):experiment.extraction_context(train,train)

    def test_binary_and_illegal_inputs(self):
        for m in [np.zeros((0,2)),np.array([[np.nan]]),np.array([[2]]),np.zeros(3)]:
            with self.assertRaises(ValueError):experiment.binary(m)
        with self.assertRaises(ValueError):experiment.local_mask(np.ones((20,20)),0,'a')

    def test_unknown_geometry_strict_majority_and_empty(self):
        bg=np.zeros((8,8),bool);bg[:4]=True
        half=np.ones((8,8),bool)
        mostly=np.zeros_like(bg);mostly[:3,:4]=True
        u,rows=experiment.unknown_union(bg,[half,mostly,np.zeros_like(bg)])
        np.testing.assert_array_equal(u,mostly);self.assertEqual([r['index'] for r in rows],[1])
        self.assertFalse((u&~bg).any())
        with self.assertRaises(ValueError):experiment.unknown_union(bg,[np.zeros((3,3))])

    def test_tiles_are_local_deterministic_and_never_cross_foreground(self):
        bg=np.ones((144,144),bool);bg[:,60:65]=False
        a,box=experiment.local_mask(bg,8,'image',min_area=20)
        b,other=experiment.local_mask(bg.copy(),8,'image',min_area=20)
        np.testing.assert_array_equal(a,b);self.assertEqual(box,other)
        self.assertFalse((a&~bg).any());self.assertLess(a.sum(),bg.sum())
        empty,box=experiment.local_mask(np.zeros_like(bg),8,'image')
        self.assertIsNone(box);self.assertFalse(empty.any())
        tiny=np.zeros_like(bg);tiny[1,1]=True
        self.assertIsNone(experiment.local_mask(tiny,8,'image')[1])

    def test_signature_method_and_parameters_are_separate(self):
        base={'format_version':3006,'experimental_pooling':{'checkpoint':'a'*64}}
        a=experiment.bank_signature(base,'split_matched',8)
        self.assertNotEqual(a,experiment.bank_signature(base,'safe_matched',8))
        self.assertNotEqual(a,experiment.bank_signature(base,'split_matched',9))
        self.assertNotEqual(a,experiment.bank_signature(dict(base,experimental_pooling={'checkpoint':'b'*64}),'split_matched',8))
        self.assertNotIn('background_protocol',base)

    @unittest.skipIf(os.name=='nt','Windows account cannot create symlinks; verify on Linux server')
    def test_matched_merge_preserves_foreground_and_excludes_same_background_images(self):
        from src.feature_bank_storage import save_feature_matrix_npz,load_feature_matrix_npz
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/'original';cache=p/'cache';attempt=p/'attempt';attempt.mkdir()
            fg=np.eye(2,1280,dtype=np.float32);bg=np.eye(2,1280,k=2,dtype=np.float32)
            save_feature_matrix_npz(raw/'features/cat/features.npz',fg,['a_000001','b_000002'])
            save_feature_matrix_npz(raw/'features/background/features.npz',bg,['a_000001','b_000002'])
            experiment.ab.write(raw/'meta/class_mapping.json',{'id_to_dirname':{'0':'background','1':'cat'}})
            job=dict(raw=str(raw),train_ids=['a','b'],cache_identity={'a':{},'b':{}},raw_signature={})
            before=experiment.ab.digest(raw/'features/cat/features.npz')
            for i,stem in enumerate(['a','b']):
                file=cache/stem/'data.npz';file.parent.mkdir(parents=True)
                np.savez(file,vectors=np.stack([bg[i],fg[0],fg[1]]))
                report=cache/stem/'regions.json'
                experiment.ab.write(report,dict(sample_id=f'{stem}_00000{i+1}',eligible=i==0,
                    unknown_candidates=[],unknown_pixels=0,original_area=400))
                experiment.ab.write(cache/stem/'complete.json',dict(identity=dict(input={},protocol=experiment.PROTOCOL,side=8,active=experiment.ACTIVE),
                    file=str(file),report=str(report),artifacts=experiment.ab.identities([file,report])))
            result,_=experiment.merge_banks(job,cache,p/'new',8,attempt)
            for root in result['raw'].values():
                members=experiment.ab.bank_members(root)
                self.assertEqual(members['background'],{'a_000001'})
                self.assertEqual(members['cat'],{'a_000001','b_000002'})
                x,ids=load_feature_matrix_npz(Path(root)/'features/cat/features.npz')
                np.testing.assert_array_equal(x,fg)
            self.assertEqual(before,experiment.ab.digest(raw/'features/cat/features.npz'))

    def test_incomplete_group_has_no_mean(self):
        s=experiment.summarize({'split_matched_42':{'miou':.8}})
        self.assertIsNone(s['groups']['split_matched']['mean']);self.assertFalse(s['contrasts'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_pool_uses_same_hard_token_gmp_formula(self):
        import torch
        from torch.nn import functional as F
        class Model:pass
        torch.manual_seed(42);tokens=torch.randn(1,5184,4)
        mask=np.zeros((1008,1008),bool);mask[0:14,0:28]=True
        got,info=experiment.pool(Model(),tokens,mask)
        wanted,_=experiment.gmp.solve(tokens[0,:2],torch.ones(2),.1)
        np.testing.assert_allclose(got,F.normalize(wanted,dim=0).numpy(),rtol=1e-6,atol=1e-6)
        self.assertEqual(info['support'],2)
        with self.assertRaises(ValueError):experiment.pool(Model(),tokens,np.zeros_like(mask))


if __name__=='__main__':unittest.main()
