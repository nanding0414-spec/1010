import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from tools import experimental_query_probe as q


class ProbeTests(unittest.TestCase):
    def test_selection_fixed_disjoint_and_covers_classes(self):
        plan={}
        for c in range(20):
            for suffix in ('a','b','v'):
                plan[f'{c}_{suffix}']=[dict(cls=f'class{c}')]
        split=dict(train_images=[s for s in plan if not s.endswith('v')],validation_images=[s for s in plan if s.endswith('v')])
        selected,coverage,classes=q.choose_images(plan,split)
        self.assertEqual(q.choose_images(plan,split)[0],selected)
        self.assertEqual(len(classes),20)
        self.assertFalse(set(selected['gallery'])&set(selected['heldout']))
        self.assertTrue(all(n==2 for n in coverage['gallery'].values()))

    def test_filter_indices_preserve_sort_ties_and_mask_cleanup(self):
        p=SimpleNamespace(min_mask_region_area=0,min_mask_area=1,max_mask_area_ratio=0,mask_nms_threshold=1.)
        masks=np.array([[[1,0]],[[0,0]],[[0,1]],[[1,1]]],bool)
        ids,arrays=q.filter_indices(p,masks,[.8,.9,.8,.7],[1,1,1,2],(1,2),lambda m,_:m,None)
        self.assertEqual(ids,[0,2,3])
        np.testing.assert_array_equal(arrays[0],masks[ids])
        p.mask_nms_threshold=.7
        ids,_=q.filter_indices(p,masks,[.8,.9,.8,.7],[1,1,1,2],(1,2),lambda m,_:m,lambda *_:[0])
        self.assertEqual(ids,[0,3])

    def test_region_query_merge_split_and_no_background_fabrication(self):
        region=np.ones((1,3),bool)
        masks=np.array([[[1,1,0]],[[0,1,1]]],bool)
        vec,diag=q.regional_query(region,1,masks,np.array([.6,.8]),np.array([1,1]),np.eye(2,dtype=np.float32))
        np.testing.assert_allclose(vec,[1/np.sqrt(5),2/np.sqrt(5)],rtol=1e-6)
        self.assertEqual(diag['pixel_counts'],[1,2])
        with self.assertRaises(ValueError):q.regional_query(region,2,masks,np.array([.6,.8]),np.array([1,1]),np.eye(2))
        with self.assertRaises(ValueError):q.normalize(np.zeros((1,3)))
        with self.assertRaises(ValueError):q.normalize(np.array([[np.nan,1]]))

    def test_matching_is_class_blind_and_reports_missing(self):
        region=np.array([[1,1,0]],bool)
        masks=np.array([[[0,0,1]],[[1,1,0]]],bool)
        self.assertEqual(q.match_full(region,masks),(1,1.))
        self.assertEqual(q.match_full(region,np.empty((0,1,3),bool)),(-1,0.))
        self.assertEqual(q.match_full(region,masks[:1]),(-1,0.))

    def test_knn_normalization_and_concat_equal_branch_weight(self):
        gallery=np.array([[1,0],[0,1]],np.float32)
        labels=np.array(['a','b'])
        np.testing.assert_array_equal(q.predict(gallery,labels,gallery),labels)
        concat=q.normalize(np.concatenate([q.normalize(gallery*100),q.normalize(gallery)],axis=1))
        np.testing.assert_allclose(np.linalg.norm(concat[:,:2],axis=1),1/np.sqrt(2),rtol=1e-6)

    def test_image_cache_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stages=q.ab.Stages(root,True)
            def action(attempt):
                file=attempt/'vectors.npz';q.write_npz(file,x=np.eye(2))
                return dict(file=str(file)),[file]
            result=stages.run('image',action)
            self.assertEqual(result,stages.run('image',lambda _:self.fail('must skip')))
            with self.assertRaises(FileExistsError):q.write_npz(result['file'],x=np.eye(2))
            Path(result['file']).write_bytes(b'changed')
            with self.assertRaises(ValueError):stages.verify_all()

    def test_summary_compares_identical_matched_subset_and_separates_prompt_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for stem in ('train','held'):
                def action(attempt):
                    path=attempt/'vectors.npz'
                    q.write_npz(path,conditional=np.eye(2,dtype=np.float32),full=np.eye(2,dtype=np.float32),
                        gmp=np.eye(2,dtype=np.float32),valid=np.array([True,False]),prompt_classes=np.array([0,-1]),
                        classes=np.array(['cat','dog']),ids=np.array([stem+'0',stem+'1']))
                    return dict(file=str(path)),[path]
                q.ab.Stages(root,True).run(stem,action)
            result=q.summarize(root,dict(gallery=['train'],heldout=['held']),['cat','dog'],['cat','dog'])
            self.assertEqual(result['full_match_coverage'],.5)
            self.assertEqual(result['retrieval']['full_gmp']['count'],1)
            self.assertEqual(result['retrieval']['full_query']['count'],1)
            self.assertEqual(result['retrieval']['full_prompt_class']['accuracy'],1.)
            self.assertIsNone(result['retrieval']['full_query']['by_class']['dog']['accuracy'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally; run on server')
class CaptureTests(unittest.TestCase):
    def test_real_tensor_query_slots_filter_and_cap(self):
        import torch
        from unittest.mock import patch
        import sys
        class P:
            confidence_threshold=.1;min_mask_region_area=0;min_mask_area=1
            max_mask_area_ratio=0;mask_nms_threshold=1.;max_masks=1
            def _ground(self,state,indices,expand):
                return dict(queries=torch.tensor([[[1.,0.],[0.,1.]]]),
                    pred_logits=torch.tensor([[[1.],[2.]]]),presence_logit_dec=torch.tensor([4.]))
            def _filter(self,masks,scores,labels,shape):
                ids=np.argsort(-np.asarray(scores),kind='stable')
                return np.stack(masks)[ids],np.asarray(scores,np.float32)[ids],np.asarray(labels,np.int32)[ids]
            def __call__(self,img,class_indices):
                out=self._ground(None,[0],True)
                scores=(out['pred_logits'].sigmoid().squeeze(-1)*out['presence_logit_dec'].sigmoid().reshape(1,1))[0].tolist()
                masks,scores,labels=self._filter([np.array([[1,0]],bool),np.array([[0,1]],bool)],scores,[1,1],(1,2))
                self.last_result=dict(class_ids=labels[:1])
                return dict(instances=SimpleNamespace(pred_masks=torch.from_numpy(masks[:1]),scores=torch.from_numpy(scores[:1])))
        modules={'src.mask_proposer_sam3':SimpleNamespace(_postprocess_small_regions=lambda m,_:m,_nms_by_masks=None)}
        with patch.dict(sys.modules,modules):
            masks,scores,labels,queries,slots=q.Capture(P())(np.zeros((1,2,3),np.uint8),[0])
        np.testing.assert_array_equal(queries,[[0,1]])
        self.assertEqual(slots[0]['query_index'],1)


if __name__=='__main__':unittest.main()
