import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools import experimental_rescue_mechanisms as r
from tests.test_experimental_overnight_rescue import row


class MechanismTests(unittest.TestCase):
    def test_strict_matches_previous_random_masks(self):
        rng=np.random.default_rng(42)
        for _ in range(40):
            strong=[row(rng.random(80)>.8)]
            extras=[row(rng.random(80)>.7, cls='chair' if i%2 else 'bottle',
                        q=float(rng.random()),retr=float(rng.random()),margin=float(rng.random()),score=float(rng.random())) for i in range(20)]
            a,n=r.rescue_rows(strong,extras,r.VARIANTS['rescue_strict'])
            b,m=r.prior.rescue_rows(strong,extras,r.prior.RECIPE['rescue_strict'])
            self.assertEqual(n,m)
            for x,y in zip(a,b):np.testing.assert_array_equal(x['mask'],y['mask'])

    def test_conflict_rejects_other_class_not_same_class(self):
        strong=[row([1,0,0])]
        extra=[row([1,1,1],cls='bottle')]
        self.assertEqual(r.rescue_rows(strong,extra,r.VARIANTS['conflict10'])[1],0)
        extra[0]['cls_name']='chair'
        self.assertEqual(r.rescue_rows(strong,extra,r.VARIANTS['conflict10'])[1],1)

    def test_votes_require_distinct_same_class_support(self):
        extras=[row([1,1,0]),row([0,1,1])]
        out,n=r.rescue_rows([],extras,r.VARIANTS['vote2'])
        self.assertEqual(n,1);np.testing.assert_array_equal(out[0]['mask'],[[0,1,0]])
        extras[1]['cls_name']='bottle'
        self.assertEqual(r.rescue_rows([],extras,r.VARIANTS['vote2'])[1],0)
        with self.assertRaises(ValueError):r.rescue_rows([],extras[:1]*2,r.VARIANTS['vote2'])

    def test_votes_exclude_failed_gates(self):
        extras=[row([1,1,0]),row([0,1,1],q=.1)]
        self.assertEqual(r.rescue_rows([],extras,r.VARIANTS['vote2'])[1],0)

    def test_conflict_uses_visible_original_class(self):
        strong=[row([1,0],cls='chair',score=.99),row([1,0],cls='bottle',score=.9)]
        self.assertEqual(r.rescue_rows(strong,[row([1,1])],r.VARIANTS['conflict10'])[1],1)

    def test_added_pixel_diagnostics_ignore_void_and_existing_pixels(self):
        gt=np.array([0,2,3,255,2]);original=np.array([0,0,0,0,2]);pred=np.array([2,2,2,2,2])
        v=r.added_pixel_counts(gt,original,pred)
        self.assertEqual(v['added_background_by_class'][2],1)
        self.assertEqual(v['recovered_by_class'][2],1)
        self.assertEqual(v['new_wrong_foreground_by_class'][2],1)

    def test_retrieval_order_changes_only_new_pixel_allocation(self):
        strong=[row([1,0,0,0])]
        extras=[row([0,1,1,0],cls='bottle',score=.99,retr=.91),row([0,0,1,1],score=.95,retr=.99)]
        a,_=r.rescue_rows(strong,extras,r.VARIANTS['rescue_strict'])
        b,_=r.rescue_rows(strong,extras,r.VARIANTS['retrieval_order'])
        self.assertEqual(a[1]['cls_name'],'bottle');self.assertEqual(b[1]['cls_name'],'chair')
        for result in (a,b):
            np.testing.assert_array_equal(result[0]['mask'],strong[0]['mask'])
            self.assertFalse((result[1]['mask']&result[2]['mask']).any())

    def test_all_variants_preserve_input_and_foreground(self):
        for params in r.VARIANTS.values():
            strong=[row([1,0,0,0])];extras=[row([1,1,0,0]),row([0,1,1,0])]
            before=copy.deepcopy(extras);out,_=r.rescue_rows(strong,extras,params)
            for a,b in zip(extras,before):np.testing.assert_array_equal(a['mask'],b['mask'])
            for added in out[1:]:self.assertFalse((added['mask']&strong[0]['mask']).any())

    def test_invalid_empty_and_zero_mask(self):
        p=r.VARIANTS['rescue_strict']
        self.assertEqual(r.rescue_rows([],[],p),([],0))
        self.assertEqual(r.rescue_rows([],[row([0,0])],p),([],0))
        with self.assertRaises(ValueError):r.rescue_rows([],[row([1],q=float('nan'))],p)

    def test_fixed_plan_replays_first_and_all_48_unique(self):
        plan=r.evaluation_plan();flat=[x for batch in plan for x in batch]
        self.assertEqual(len(flat),48);self.assertEqual(len(set(flat)),48)
        self.assertEqual([x[0] for x in plan[0]],['baseline']*3)
        self.assertEqual([x[0] for x in plan[1]],['rescue_strict']*3)

    def test_stage_resume_checks_hash_and_uses_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            stages=r.ab.Stages(Path(directory),True);calls=[]
            def action(attempt):
                calls.append(attempt);file=attempt/'value.txt';file.write_text('ok')
                return {'file':str(file)},[file]
            first=stages.run('one',action)
            self.assertEqual(stages.run('one',action),first);self.assertEqual(len(calls),1)
            Path(first['file']).write_text('changed')
            with self.assertRaises(ValueError):stages.run('one',action)

    def test_local_failure_continues_and_deadline_stops(self):
        seen=[];failed=[]
        def action(item,gpu):
            if item==1:raise r.queue.TaskFailure('local')
            seen.append(item)
        with patch.object(r.queue,'ensure_resources',return_value=None):
            r.queue.run_round([1,2,3],[0,1],action,threading.Event(),float('inf'),lambda i,e:failed.append(i))
        self.assertEqual(sorted(seen),[2,3]);self.assertEqual(failed,[1])
        with patch.object(r.queue,'ensure_resources',side_effect=r.queue.Deferred('deadline')):
            with self.assertRaises(r.queue.Deferred):r.queue.run_round([4],[0,1],action,threading.Event(),0,lambda *_:None)
        self.assertNotIn(4,seen)


if __name__=='__main__':unittest.main()
