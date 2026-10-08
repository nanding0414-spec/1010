import copy
import importlib.util
import unittest
import numpy as np
from tools.experimental_tag_constraint_queue import parse_tags,allowed_rows,coverage,absent_loss,summarize


class TagTests(unittest.TestCase):
    def test_tags_and_background_and_row_order(self):
        tags=parse_tags(dict(images=[dict(img_id='a',labels=[1]),dict(img_id='b',labels=[2])]),['a','b'],{'0':'background','1':'cat','2':'dog'})
        got=allowed_rows(['b','a'],['dog','background','cat'],tags)
        np.testing.assert_array_equal(got,[[True,True,False],[False,True,True]])
        c=coverage(['a','b'],[1,0],['background','cat','dog'],tags)
        self.assertEqual(c['missing_positive_by_image'],{'b':['dog']})
        self.assertEqual(len(c['pseudo_class_outside_tags']),0)

    def test_duplicate_missing_or_invalid_tags_rejected(self):
        mapping={'0':'background','1':'cat'}
        with self.assertRaises(ValueError):parse_tags(dict(images=[]),['a'],mapping)
        with self.assertRaises(ValueError):parse_tags(dict(images=[dict(img_id='a',labels=[2])]),['a'],mapping)
        with self.assertRaises(ValueError):parse_tags(dict(images=[dict(img_id='a',labels=[True])]),['a'],mapping)
        with self.assertRaises(ValueError):parse_tags(dict(images=[dict(img_id='a',labels=[1])]*2),['a'],mapping)

    def test_incomplete_mean_not_reported(self):
        s=summarize({'control_42':dict(miou=.8)})
        self.assertIsNone(s['groups']['control']['mean'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_extreme_logits_and_absent_gradient(self):
        import torch
        z=torch.tensor([[1000.,-1000.,0.]],requires_grad=True)
        loss=absent_loss(z,torch.tensor([[False,True,True]]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(z.grad[0,0],0)
        self.assertLess(z.grad[0,2],0)
        all_allowed=absent_loss(z,torch.ones_like(z,dtype=torch.bool))
        self.assertEqual(float(all_allowed.detach()),0)
        with self.assertRaises(ValueError):absent_loss(z,torch.zeros_like(z,dtype=torch.bool))
        with self.assertRaises(ValueError):absent_loss(z*float('nan'),torch.ones_like(z,dtype=torch.bool))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_zero_coefficient_short_training_is_bitwise_equal(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(42);a=torch.nn.Linear(4,3);b=copy.deepcopy(a)
        x=torch.randn(5,4);y=torch.tensor([0,1,2,0,1]);mask=torch.tensor([[True,True,False]]*5)
        oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
        for _ in range(5):
            la=F.cross_entropy(a(x),y);lb=F.cross_entropy(b(x),y)
            auxiliary=absent_loss(b(x),mask)
            self.assertTrue(torch.isfinite(auxiliary))
            oa.zero_grad();ob.zero_grad();la.backward();lb.backward();oa.step();ob.step()
        for u,v in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(u,v))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable locally')
    def test_real_batch_binding_and_best_classifier_sidecar(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from torch import nn
        from torch.nn import functional as F
        import src.adapter_training as at
        from tools.experimental_tag_constraint_queue import install_training
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'meta').mkdir()
            (p/'meta/class_mapping.json').write_text(json.dumps({'id_to_dirname':{'0':'background','1':'cat','2':'dog'}}))
            tags=p/'tags.json';tags.write_text(json.dumps({'images':[{'img_id':'a','labels':[1]},{'img_id':'b','labels':[2]}]}))
            data=(np.ones((4,1280),np.float32),np.array([0,1,0,2]),np.array(['a','a','b','b']),['background','cat','dog'],{},[])
            job=dict(raw=str(p),weights=str(p),attempt=str(p),imagelevel_json=str(tags),train_ids=['a','b'],coefficient=.1)
            with patch.object(at,'load_training_bank',return_value=data):
                state,original=install_training(job)
                try:
                    at.load_training_bank(p,'unused')
                    rows=at.balanced_batch(data[1],data[2],np.arange(4),np.random.default_rng(42),3,1)
                    head=nn.Linear(256,3);logits=head(torch.ones(len(rows),256));loss=F.cross_entropy(logits,torch.tensor(data[1][rows]))
                    loss.backward()
                    torch.save(dict(format_version=1,source_signature={},epoch=7),p/'adapter.tmp')
                    saved=torch.load(p/'classifier.pt',weights_only=True)
                    self.assertEqual(saved['epoch'],7);self.assertEqual(saved['classes'],data[3])
                    self.assertEqual(state['batches'],1);self.assertTrue(torch.equal(saved['state_dict']['weight'],head.weight.detach()))
                finally:at.load_training_bank,at.balanced_batch,nn.Linear.__init__,F.cross_entropy,torch.save=original


if __name__=='__main__':unittest.main()
