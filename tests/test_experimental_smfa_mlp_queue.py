"""Synthetic tests. Tensor/checkpoint tests require torch; no server accuracy claims."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from tools import experimental_smfa_mlp_queue as q


def metrics(signature=None):
    return dict(evaluated_ids=[f'i{i}' for i in range(1449)], evaluated_count=1449,
        eval_limit=0, cleanup=None, proposal_count_before=3609, proposal_count_after=3609,
        confusion_matrix=np.eye(21, dtype=int).tolist(), mIoU=1.,
        per_class_iou={str(i): 1. for i in range(21)}, inference_settings={'INF_TOPK_NEIGH': 25},
        proposals_dir='fixed-clean', feature_signature=signature or {'format_version': 3001})


class ProtocolTests(unittest.TestCase):
    def test_fixed_grid_and_parameter_control(self):
        self.assertEqual(len(q.NEW) * len(q.SEEDS), 21)
        self.assertNotIn('d3', q.NEW)
        self.assertEqual(q.parameter_count(q.RECIPES['d3']), 1049856)
        self.assertEqual(q.parameter_count(q.RECIPES['res4']), q.parameter_count(q.RECIPES['d4']))
        self.assertLess(abs(q.parameter_count(q.RECIPES['wide3']) / q.parameter_count(q.RECIPES['d5']) - 1), .003)
        with self.assertRaises(ValueError):
            q.Architecture(6)
        with self.assertRaises(ValueError):
            q.adapter_tag('unknown')

    def test_argv_changes_only_depth_width_seed(self):
        original = ['trainer', '--seed', '42', '--split-file', 'fixed', '--num-layers', '3',
                    '--hidden-dim', '512', '--epochs', '50', '--lr', '.001']
        result = q.training_argv(original, 123, 'wide3')
        self.assertEqual(result, ['trainer', '--seed', '123', '--split-file', 'fixed', '--num-layers', '3',
                                '--hidden-dim', '704', '--epochs', '50', '--lr', '.001'])
        self.assertEqual(original[2], '42')
        for seed, variant in [(42, 'd3'), (99, 'd4')]:
            with self.assertRaises(ValueError):
                q.training_argv(original, seed, variant)
        with self.assertRaises(ValueError):
            q.training_argv(['--seed', '42'], 42, 'd4')

    def test_build_and_reference_training_refused_before_torch(self):
        for job in [dict(stage='build', group='SMFA'), dict(stage='train', group='SMFA', variant='d3'),
                    dict(stage='eval', group='C', variant='d4')]:
            with self.assertRaises(ValueError):
                q.worker(job)

    def test_full_metrics_and_protocol_validation(self):
        value = metrics()
        q.validate_metrics(value, value, value['evaluated_ids'], 3609)
        for key, replacement in [('cleanup', {}), ('evaluated_count', 1448), ('mIoU', .9),
                                 ('proposal_count_after', 3608), ('proposals_dir', 'different'),
                                 ('inference_settings', {'INF_TOPK_NEIGH': 5})]:
            bad = copy.deepcopy(value)
            bad[key] = replacement
            with self.assertRaises(ValueError, msg=key):
                q.validate_metrics(bad, value, value['evaluated_ids'], 3609)

    def test_partial_summary_never_marked_complete(self):
        reports = {'d4_42': dict(seed=42, candidate_miou=.85, delta_pp=.1)}
        result = q.summarize(reports)
        self.assertFalse(result['d4']['complete'])
        self.assertIsNone(result['d4']['sample_std_miou'])
        self.assertIsNone(result['d5']['mean_miou'])
        for seed in (123, 2026):
            reports[f'd4_{seed}'] = dict(seed=seed, candidate_miou=.85, delta_pp=.1)
        self.assertTrue(q.summarize(reports)['d4']['complete'])

    def test_paired_structure_contrasts(self):
        reports = {}
        for seed in q.SEEDS:
            reports[f'd4_{seed}'] = dict(candidate_miou=.85)
            reports[f'res4_{seed}'] = dict(candidate_miou=.852)
        contrast = q.paired_contrasts(reports)['res4-d4']
        self.assertTrue(contrast['complete'])
        self.assertAlmostEqual(contrast['mean_delta_pp'], .2)
        self.assertFalse(q.paired_contrasts(reports)['d5-wide3']['complete'])

    def test_deadline_disk_and_cancellation(self):
        with patch.object(q.queue.time, 'time', return_value=10), \
             patch.object(q.queue.shutil, 'disk_usage', return_value=SimpleNamespace(free=q.queue.FLOOR)):
            q.queue.ensure_resources(11)
            with self.assertRaises(q.queue.Deferred):
                q.queue.ensure_resources(10)
        with patch.object(q.queue.shutil, 'disk_usage', return_value=SimpleNamespace(free=q.queue.FLOOR - 1)):
            with self.assertRaises(q.queue.Deferred):
                q.queue.ensure_resources(float('inf'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cancelled = threading.Event()
            cancelled.set()
            with patch.object(q.subprocess, 'Popen') as popen:
                with self.assertRaises(q.queue.GlobalFailure):
                    q.launch({}, root, {}, root / 'log', cancelled, 999)
                popen.assert_not_called()

    def test_only_owned_process_terminated_on_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = Mock(pid=12345)
            process.poll.return_value = None
            cancel = Mock()
            cancel.is_set.return_value = False
            cancel.wait.return_value = True
            with patch.object(q.queue, 'ensure_resources'), \
                 patch.object(q.subprocess, 'check_output', return_value='16, 0'), \
                 patch.object(q.subprocess, 'Popen', return_value=process), \
                 patch.object(q.os, 'killpg', create=True) as killpg:
                with self.assertRaises(q.queue.GlobalFailure):
                    q.launch(dict(variant='d4', seed=42, stage='train'), root,
                             {'CUDA_VISIBLE_DEVICES': '1'}, root / 'log', cancel, 999)
            if q.os.name == 'posix':
                killpg.assert_called_once_with(12345, q.signal.SIGTERM)
            else:
                process.terminate.assert_called_once()
            process.wait.assert_called_once()

    def test_cached_mask_order_and_changed_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = np.zeros((2, 3, 3), np.uint8)
            masks[0, 0, 0] = 1
            masks[1, 1, 1] = 1
            file = root / 'vectors.npz'
            np.savez_compressed(file, SMFA=np.ones((2, 1280), np.float32))
            identity = {'space': 'SMFA'}
            q.ab.write(root / 'complete.json', dict(identity=identity, mask_hashes=q.queue.mask_hashes(masks),
                file=str(file), artifacts=q.ab.identities([file])))
            self.assertEqual(q.queue.cache_vectors(root, identity, 'SMFA', q.queue.mask_hashes(masks)).shape, (2, 1280))
            with self.assertRaises(q.queue.GlobalFailure):
                q.queue.cache_vectors(root, identity, 'SMFA', q.queue.mask_hashes(masks[::-1]))
            with self.assertRaises(q.queue.GlobalFailure):
                q.queue.cache_vectors(root, {'space': 'GMP'}, 'SMFA', q.queue.mask_hashes(masks))
            file.write_bytes(b'changed')
            with self.assertRaises(q.queue.GlobalFailure):
                q.queue.cache_read(root, identity)


class EndToEndSimulation(unittest.TestCase):
    def run_case(self, root, failure=None, resume=False, calls=None):
        calls = calls if calls is not None else []
        raw = root / 'old_raw'
        raw.mkdir(exist_ok=True)
        refs = {}
        split = dict(train_images=['train'], validation_images=['heldout'])
        for seed in q.SEEDS:
            w, b = root / f'old_weights_{seed}', root / f'old_bank_{seed}'
            q.ab.write(w / 'training_report.json', split)
            q.ab.write(b / 'meta/feature_signature.json', {'format_version': 3001})
            path = root / f'old_metrics_{seed}.json'
            q.ab.write(path, metrics())
            refs[seed] = dict(raw=str(raw), weights=str(w), bank=str(b), metrics=str(path), per_image='old.npz')
        common = {k: str(root / k) for k in ('dataset_root', 'image_dir', 'checkpoint', 'train_list',
            'val_list', 'imagelevel_json', 'gt_dir', 'cache')}
        common.update(images={}, train_ids=['train'], val_ids=metrics()['evaluated_ids'], cache_identity={},
                      checkpoint_sha256='a'*64, raw_signature={'format_version': 3001}, proposals='fixed-clean')

        def launch(job, attempt, env, log, cancel, deadline):
            key = (job['variant'], job['seed'], job['stage'])
            calls.append(key)
            if failure:
                failure(key)
            q.ab.write(attempt / 'queue_worker.json', dict(status='complete'))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('simulated', encoding='utf-8')
            if job['stage'] == 'train':
                q.ab.write(Path(job['weights']) / 'training_report.json', split)
                q.ab.write(attempt / 'effective_training_argv.json',
                           dict(seed=job['seed'], architecture=q.adapter_tag(job['variant'])))
                q.ab.write(Path(job['bank']) / 'meta/feature_signature.json', dict(adapter=dict(type=q.FORMAT,
                    architecture=dict(experimental_mlp=q.adapter_tag(job['variant'])))))
            else:
                sig = q.ab.read(Path(job['bank']) / 'meta/feature_signature.json')
                q.ab.write(Path(job['metrics']), metrics(sig))
                (attempt / 'per_image_confusion.npz').write_bytes(b'simulation')

        with patch.object(q, 'ROOT', root), patch.object(q, 'verified_history', return_value=(common, refs, split, [], {'count': 3609})), \
             patch.object(q.importlib.util, 'find_spec', return_value=object()), patch.object(q.ab, 'check_gpu'), \
             patch.object(q.ab, 'source_identity', return_value=[]), patch.object(q.ab, 'bank_members', return_value={'background': {'one'}}), \
             patch.object(q, 'launch', side_effect=launch), patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=30*2**30)), \
             patch.object(q.seeds, 'validate_image_matrices'), patch.object(q.os, 'chdir'):
            flags = ['--version', 'test_mlp', '--previous', str(root / 'previous')]
            if resume:
                flags.append('--resume')
            q.main(flags)
        return calls, q.ab.read(root / 'experiments/test_mlp/summary.json')

    def test_replay_before_21_trainings_and_resume_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, report = self.run_case(root)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['results']), 24)
            self.assertEqual(len([c for c in calls if c[2] == 'train']), 21)
            self.assertEqual(set(calls[:3]), {('d3', s, 'eval') for s in q.SEEDS})
            replay_calls, resumed = self.run_case(root, resume=True)
            self.assertEqual(replay_calls, [])
            self.assertEqual(resumed['status'], 'complete')
            with self.assertRaises(ValueError):
                self.run_case(root)

    def test_global_replay_failure_prevents_all_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def fail(key):
                if key[1] == 42:
                    raise q.queue.GlobalFailure('Historical mismatch')
            with self.assertRaises(q.queue.GlobalFailure):
                self.run_case(Path(tmp), failure=fail, calls=calls)
            self.assertFalse(any(c[2] == 'train' for c in calls))

    def test_local_retry_once_then_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(key):
                if key == ('d4', 42, 'train'):
                    raise q.queue.TaskFailure('Synthetic training failure')
            calls, report = self.run_case(Path(tmp), failure=fail)
            self.assertEqual(calls.count(('d4', 42, 'train')), 2)
            self.assertEqual(report['status'], 'complete_with_failures')
            self.assertEqual(len(report['results']), 23)
            self.assertIn('wide3_2026', report['results'])
            attempts = list((Path(tmp) / 'experiments/test_mlp/train_d4_42').glob('attempt_*/failed.json'))
            self.assertEqual(len(attempts), 2)

    def test_oom_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(key):
                if key == ('d5', 42, 'train'):
                    raise q.queue.TaskOOM('Synthetic OOM')
            calls, report = self.run_case(Path(tmp), failure=fail)
            self.assertEqual(calls.count(('d5', 42, 'train')), 1)
            self.assertEqual(report['status'], 'complete_with_failures')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; run in moduseg_sam3')
class TensorTests(unittest.TestCase):
    def test_plain_d3_matches_original_initialization_and_output(self):
        import torch
        from src.feature_adapter import FeatureAdapter
        cls = q.make_adapter_class(FeatureAdapter)
        torch.manual_seed(42)
        baseline = FeatureAdapter(1280, 512, 256, 3)
        torch.manual_seed(42)
        candidate = cls(1280, 512, 256, 3, q.adapter_tag('d3'))
        for k, v in baseline.state_dict().items():
            self.assertTrue(torch.equal(v, candidate.state_dict()[k]))
        x = torch.randn(5, 1280)
        self.assertTrue(torch.equal(baseline(x), candidate(x)))

    def test_all_shapes_parameters_gradients_empty_vectors_and_invalid(self):
        import torch
        from src.feature_adapter import FeatureAdapter, project_features
        cls = q.make_adapter_class(FeatureAdapter)
        for variant, recipe in q.RECIPES.items():
            model = cls(1280, recipe.width, 256, recipe.depth, q.adapter_tag(variant))
            self.assertEqual(sum(p.numel() for p in model.parameters()), q.parameter_count(recipe))
            x = torch.randn(4, 1280)
            y = model(x)
            self.assertEqual(tuple(y.shape), (4, 256))
            self.assertTrue(torch.allclose(y.norm(dim=1), torch.ones(4), atol=1e-6))
            y[:, 0].sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(np.array_equal(project_features(model, np.zeros((1, 1280), np.float32), 'cpu'), np.zeros((1, 256))))
            with self.assertRaises(ValueError):
                model(torch.full((1, 1280), float('nan')))

    def test_checkpoint_roundtrip_rejects_wrong_architecture_and_source(self):
        import torch
        import src.feature_adapter as fa
        originals = (fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature)
        try:
            q.install_adapter('res4')
            model = fa.FeatureAdapter(1280, 512, 256, 4)
            source = dict(backbone='sam3', embed_dim=1280, experimental_pooling={'mode': 'smfa'})
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'adapter.pt'
                payload = dict(format_version=1, source_signature=source,
                               architecture=model.architecture, state_dict=model.state_dict())
                torch.save(payload, path)
                restored, _ = fa.load_adapter(path, 'cpu', source)
                x = torch.randn(3, 1280)
                self.assertTrue(torch.equal(model(x), restored(x)))
                with self.assertRaises(ValueError):
                    fa.load_adapter(path, 'cpu', dict(source, experimental_pooling={'mode': 'gmp'}))
                payload['architecture']['experimental_mlp'] = q.adapter_tag('d4')
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    fa.load_adapter(path, 'cpu', source)
                with self.assertRaises(TypeError):
                    originals[0](**model.architecture)
        finally:
            fa.FeatureAdapter, fa.load_adapter, fa.adapted_signature = originals


if __name__ == '__main__':
    unittest.main()
