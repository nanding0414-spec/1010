import copy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from tools import experimental_sam3_semantic_queue as m


class SemanticTests(unittest.TestCase):
    def test_spatial_class_alignment_and_chunking(self):
        q=np.zeros((3,4,4),bool); q[0,:2]=1; q[1,2:]=1
        c=np.stack([q[0],q[0],q[1],np.ones((4,4),bool)])
        a=m.semantic_evidence(q,c,[1,2,1,3],[.8,.9,.7,1.])
        b=m.semantic_evidence(q,c,[1,2,1,3],[.8,.9,.7,1.],chunk=1)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(a[0][:2,:3],[[.8,.9,.5],[.7,0,.5]])
        self.assertFalse(a[0][2].any())

    def test_absent_candidates_and_tiny_masks(self):
        q=np.ones((1,1,1),bool)
        self.assertFalse(m.semantic_evidence(q,np.zeros((0,1,1),bool),[],[])[0].any())
        self.assertEqual(m.semantic_evidence(q,q,[20],[1])[0][0,19],1)
        self.assertEqual(m.semantic_evidence(np.zeros((0,1,1),bool),q,[1],[1])[0].shape,(0,20))

    def test_invalid_inputs(self):
        q=np.ones((1,2,2),bool)
        for classes,scores in [([0],[.5]),([21],[.5]),([1.1],[.5]),([1],[np.nan]),([1],[1.01])]:
            with self.assertRaises(ValueError):m.semantic_evidence(q,q,classes,scores)
        with self.assertRaises(ValueError):m.semantic_evidence(q,np.ones((1,3,3)),[1],[1])
        with self.assertRaises(ValueError):m.semantic_evidence(q*2,q,[1],[1])

    def test_decisions_preserve_no_match_and_reordered_labels(self):
        names=list(reversed(m.provenance.evidence.VOC_CLASSES)); scores=np.zeros(21,np.float32)
        scores[names.index('chair')]=.8; scores[names.index('sofa')]=.78
        before=scores.copy(); sem=np.zeros(20,np.float32); sem[17]=.9
        for method in ('rank_bonus','guarded'):
            self.assertEqual(m.decide(scores,sem,names,method)[0],'sofa')
            self.assertEqual(m.decide(scores,np.zeros(20),names,method)[0],'chair')
        self.assertEqual(m.decide(scores,np.zeros(20),names,'sam_label')[0],'background')
        self.assertEqual(m.decide(scores,sem,names,'baseline')[0],'chair')
        scores[names.index('chair')]=.95
        self.assertEqual(m.decide(scores,sem,names,'guarded')[0],'chair')
        np.testing.assert_array_equal(before[[i for i,n in enumerate(names) if n!='chair']],scores[[i for i,n in enumerate(names) if n!='chair']])

    def test_guard_rejects_ambiguous_semantics_and_stable_tie(self):
        names=list(m.provenance.evidence.VOC_CLASSES); s=np.zeros(21);s[9]=.8;s[18]=.79
        e=np.zeros(20);e[17]=.8;e[8]=.75
        self.assertEqual(m.decide(s,e,names,'guarded')[0],'chair')
        e[:]=1
        self.assertEqual(m.decide(s,e,names,'sam_label')[0],'aeroplane')
        e[0]=np.nan
        with self.assertRaises(ValueError):m.decide(s,e,names,'rank_bonus')

    def test_cache_rejects_order_content_and_identity(self):
        with tempfile.TemporaryDirectory() as t:
            job=dict(semantic_cache=t,cache_identity={'a':{'image_sha256':'x'}},checkpoint_sha256='w',
                source={'resolution':1008},active_classes=[],semantic_source_sha='code')
            folder=Path(t)/'a';folder.mkdir();file=folder/'x.npz'
            masks=np.zeros((2,2,2),bool);masks[0,0,0]=1;masks[1,1,1]=1
            np.savez(file,evidence=np.zeros((2,20),np.float32))
            m.ab.write(folder/'complete.json',dict(identity=m.semantic_identity(job,'a'),file=str(file),
                mask_hashes=m.queue.mask_hashes(masks),artifacts=m.ab.identities([file])))
            self.assertEqual(m.read_semantic(job,'a',masks).shape,(2,20))
            with self.assertRaises(m.queue.GlobalFailure):m.read_semantic(job,'a',masks[::-1])
            altered=copy.deepcopy(job);altered['checkpoint_sha256']='other'
            with self.assertRaises(m.queue.GlobalFailure):m.read_semantic(altered,'a',masks)
            file.write_bytes(b'broken')
            with self.assertRaises(m.queue.GlobalFailure):m.read_semantic(job,'a',masks)


class InfrastructureTests(unittest.TestCase):
    def test_identity_dedup_and_conflicts(self):
        a=dict(path=str(Path('x').resolve()),sha256='a',size=2)
        self.assertEqual(m.unique_records([a,a]),[a])
        with self.assertRaises(ValueError):m.unique_records([a,dict(a,sha256='b')])

    def test_partial_summary_does_not_claim_average(self):
        result=m.summarize({'baseline_42':dict(miou=.85),'rank_bonus_42':dict(miou=.86)})
        self.assertIsNone(result['groups']['rank_bonus']['mean'])
        self.assertFalse(result['contrasts'])

    def test_stages_do_not_overwrite_and_retry_new_attempt(self):
        with tempfile.TemporaryDirectory() as t:
            stage=m.AliasAwareStages(t,True);attempts=[]
            def fail(a):attempts.append(a);raise RuntimeError('local')
            with self.assertRaises(RuntimeError):stage.run('a',fail)
            def good(a):
                attempts.append(a);m.ab.write(a/'x.json',{'a':1});return {'ok':True},[a]
            self.assertEqual(stage.run('a',good),{'ok':True})
            self.assertEqual([p.name for p in attempts],['attempt_001','attempt_002'])
            stage.run('a',lambda _:self.fail('completed stage relaunched'))
            (attempts[-1]/'x.json').write_text('{}')
            with self.assertRaises(ValueError):stage.run('a',good)

    def test_two_lanes_exclusive_local_failure_continues(self):
        active=set();guard=threading.Lock();done=[];fail=[]
        def action(item,gpu):
            with guard:
                self.assertNotIn(gpu,active);active.add(gpu)
            try:
                time.sleep(.005)
                if item==1:raise m.queue.TaskFailure('local')
                done.append(item)
            finally:
                with guard:active.remove(gpu)
        with patch.object(m.queue,'ensure_resources'):
            m.queue.run_round(list(range(6)),[0,1],action,threading.Event(),float('inf'),lambda i,e:fail.append(i))
        self.assertEqual(sorted(done),[0,2,3,4,5]);self.assertEqual(fail,[1])

    def test_global_failure_and_deadline_stop(self):
        def fail(*_):raise m.queue.GlobalFailure('corrupt cache')
        with patch.object(m.queue,'ensure_resources'):
            with self.assertRaises(m.queue.GlobalFailure):m.queue.run_round([1],[0,1],fail,threading.Event(),float('inf'),lambda *_:None)
        with patch.object(m.queue,'ensure_resources',side_effect=m.queue.Deferred('deadline')):
            with self.assertRaises(m.queue.Deferred):m.queue.run_round([1],[0,1],lambda *_:self.fail('launched'),threading.Event(),0,lambda *_:None)


if __name__=='__main__':unittest.main()
