"""Geometry and protocol tests; synthetic tensors do not establish model accuracy."""
import copy
from collections import Counter
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tools import experimental_mpsa_ab as m


class GeometryTests(unittest.TestCase):
    def test_empty_full_and_exact_center(self):
        indices, info = m.selection(np.zeros((1008, 1008), bool), (0, 0))
        self.assertEqual(len(indices), 0); self.assertTrue(info['empty'])
        indices, info = m.selection(np.ones((1008, 1008), bool), (7, 7))
        self.assertEqual(len(indices), 5184); self.assertFalse(info['nearest_fallback'])
        mask = np.zeros((1008, 1008), bool); mask[7, 21] = True
        np.testing.assert_array_equal(m.selection(mask, (0, 0))[0], [1])

    def test_shift_direction_and_non_square_original(self):
        mask = np.zeros((504, 1008), bool); mask[0, 0] = True
        index, info = m.selection(mask, (7, 7))
        np.testing.assert_array_equal(index, [0]); self.assertFalse(info['nearest_fallback'])
        # Positive shift means the center samples center-minus-offset.
        mask = np.zeros((1008, 1008), bool); mask[7, 14] = True
        np.testing.assert_array_equal(m.selection(mask, (7, 0))[0], [1])

    def test_thin_tiny_and_clipped_centroid_fallback(self):
        mask = np.zeros((1008, 1008), bool); mask[-1, -1] = True
        for offset in m.OFFSETS:
            index, info = m.selection(mask, offset)
            np.testing.assert_array_equal(index, [5183]); self.assertTrue(info['nearest_fallback'])
        mask[:] = False; mask[:, 2] = True
        index, info = m.selection(mask, (0, 0))
        self.assertEqual(len(index), 1); self.assertTrue(info['nearest_fallback'])

    def test_binary_validation_and_parameters(self):
        for mask in (np.array([[.5]]), np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((0, 4)), np.zeros(5)):
            with self.assertRaises(ValueError): m.selection(mask, (0, 0))
        for value in (0, 3, True, 1.0):
            with self.assertRaises(ValueError): m.MPSA(value, 'a'*64)
        with self.assertRaises(ValueError): m.MPSA(4)
        with self.assertRaises(ValueError): m.selection(np.ones((2, 2)), (-7, 0))


class ProtocolTests(unittest.TestCase):
    def test_only_reviewed_visualization_addition_allowed(self):
        original = m.ROOT/'tools/visualize_proposal_gt_diff.py'
        content = original.read_text(encoding='utf-8-sig')
        with tempfile.TemporaryDirectory() as folder, patch.object(m, 'ROOT', Path(folder)):
            file = Path(folder)/'tools/visualize_proposal_gt_diff.py'
            file.parent.mkdir(); file.write_bytes(content.replace('\n', '\r\n').encode('utf-8'))
            other = Path(folder)/'src/new_model.py'
            rows = [{'path': str(file)}, {'path': str(other)}]
            self.assertEqual(m.baseline_source_view(rows), [rows[1]])
            file.write_text(content+'\n# unreviewed change\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different content'): m.baseline_source_view(rows)

    def test_baseline_inventory_restored_on_success_and_failure(self):
        records = [{'path': 'unchanged-production.py'}]
        def check(_):
            self.assertEqual(m.ab.source_identity(), records)
            return 'verified'
        with patch.object(m.ab, 'source_identity', return_value=records) as full, \
             patch.object(m, 'baseline_source_view', side_effect=lambda rows: rows), \
             patch.object(m.rpg, 'baseline', side_effect=check):
            self.assertEqual(m.checked_baseline(Path('old')), 'verified')
            self.assertIs(m.ab.source_identity, full)
            with patch.object(m.rpg, 'baseline', side_effect=ValueError('old SHA mismatch')):
                with self.assertRaisesRegex(ValueError, 'old SHA mismatch'): m.checked_baseline(Path('old'))
            self.assertIs(m.ab.source_identity, full)

    def test_signatures_reject_phases_weights_and_production(self):
        from src import feature_signature as fs
        base = {key: None for key in fs.REQUIRED_SIGNATURE_KEYS}
        base['format_version'] = 2
        expected = m.signature(base, 'mpsa', m.MPSA(4, 'a'*64))
        with tempfile.TemporaryDirectory() as folder, patch.object(fs, 'REQUIRED_SIGNATURE_KEYS', fs.REQUIRED_SIGNATURE_KEYS | {'experimental_pooling'}):
            fs.save_feature_signature(folder, expected)
            fs.validate_feature_signature(folder, expected)
            for other in (base, m.signature(base, 'mpsa', m.MPSA(1, 'a'*64)),
                          m.signature(base, 'mpsa', m.MPSA(4, 'b'*64)), dict(expected, adapter={'sha256': 'different'})):
                with self.assertRaises(RuntimeError): fs.validate_feature_signature(folder, other)

    def test_no_second_cleanup_cli(self):
        original = ['--strict-eval', '--cleanup-config', 'old.json', '--eval-limit', '0']
        self.assertEqual(m.clean_eval_argv(original), ['--strict-eval', '--eval-limit', '0'])
        self.assertIn('--cleanup-config', original)
        with self.assertRaises(ValueError): m.clean_eval_argv(['--strict-eval'])

    def test_replay_refuses_different_pixels_or_recleaning(self):
        old = dict(evaluated_ids=['a'], proposal_count_after=3, confusion_matrix=[[1, 2], [3, 4]])
        now = dict(old, cleanup=None, proposal_count_before=3)
        m.require_baseline_replay(old, now)
        for changed in (dict(now, cleanup=m.CLEAN), dict(now, confusion_matrix=[[2, 1], [3, 4]]),
                        dict(now, evaluated_ids=['b']), dict(now, proposal_count_before=4)):
            with self.assertRaises(ValueError): m.require_baseline_replay(old, changed)

    def test_clean_cache_provenance_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)/'raw'; clean = Path(folder)/'clean'; raw.mkdir(); clean.mkdir()
            masks = np.ones((2, 5, 6), np.uint8); scores = np.array([.9, .8], np.float32)
            np.savez(raw/'a.npz', masks=masks, scores=scores)
            np.savez(clean/'a.npz', masks=masks[:1], scores=scores[:1], source_indices=np.array([0]))
            meta = dict(input_dir=str(raw), config=m.CLEAN, errors=0, total=1,
                        files=[dict(name='a.npz', source_sha256=m.ab.digest(raw/'a.npz'))])
            m.ab.write(clean/'cleanup_config.json', meta)
            before = m.ab.digest(raw/'a.npz')
            result = m.validate_clean_cache(clean, ['a'], raw, 1)
            self.assertEqual(result['count'], 1); self.assertEqual(before, m.ab.digest(raw/'a.npz'))
            with self.assertRaises(ValueError): m.validate_clean_cache(clean, ['a'], raw, 2)
            meta['config'] = dict(m.CLEAN, min_score=.8); m.ab.write(clean/'cleanup_config.json', meta)
            with self.assertRaises(ValueError): m.validate_clean_cache(clean, ['a'], raw, 1)

    def test_stage_failure_new_attempt_resume_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stages = m.ab.Stages(root, False)
            def fail(attempt):
                (attempt/'partial').write_text('retained')
                raise ValueError('simulated failure')
            with self.assertRaises(ValueError): stages.run('extract', fail)
            self.assertTrue((root/'extract/attempt_001/partial').exists())
            seen = []
            def success(attempt):
                seen.append(attempt.name); f = attempt/'result'; f.write_text('ok')
                return {'ok': True}, [f]
            self.assertEqual(m.ab.Stages(root, True).run('extract', success), {'ok': True})
            self.assertEqual(seen, ['attempt_002'])
            m.ab.Stages(root, True).run('extract', lambda _: self.fail('Must resume'))
            (root/'extract/attempt_002/result').write_text('changed')
            with self.assertRaises(ValueError): m.ab.Stages(root, True).run('extract', success)

    def test_pipeline_order_shared_inputs_and_resume(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); previous_cwd = Path.cwd()
            stack.callback(os.chdir, previous_cwd)
            checkpoint = root/'checkpoint'; checkpoint.write_text('weights')
            base_weights = root/'old_weights'
            split = dict(train_images=['t1', 't2'], validation_images=['held_out'])
            m.ab.write(base_weights/'training_report.json', split)
            matrix = np.eye(21, dtype=int).tolist()
            metric = dict(evaluated_ids=['v'], cleanup=None, proposal_count_before=3,
                          proposal_count_after=3, confusion_matrix=matrix)
            historical = root/'historical.json'; m.ab.write(historical, metric)
            base = dict(weights=str(base_weights), bank=str(root/'old_bank'), raw=str(root/'old_raw'), metrics=str(historical))
            options = {k: str(root/k) for k in ('dataset_root', 'image_dir', 'imagelevel_json', 'train_list', 'val_list', 'gt_dir')}
            options['checkpoint'] = str(checkpoint)
            old = dict(options=options, selected=dict(pseudo=str(root/'pseudo'), proposals=str(root/'raw_proposals')))
            args = SimpleNamespace(**{k: Path(v) for k, v in options.items()})
            context = dict(selected=dict(pseudo=str(root/'pseudo'), proposals=str(root/'clean')),
                           inputs=[], code=[], images={'t1': 't1.jpg', 't2': 't2.jpg', 'v': 'v.jpg'},
                           train_ids=['t1', 't2'], val_ids=['v'])
            # Include the metadata hash in the same way as a real launch.
            m.ab.write(root/'clean/cleanup_config.json', {})
            launched = []
            def launch(job, attempt, env, log, cancel=None):
                launched.append((job['stage'], job['mode'], job.get('pmfa', {}).get('phases')))
                log.parent.mkdir(parents=True, exist_ok=True); log.write_text('worker')
                m.ab.write(attempt/'worker_report.json', dict(status='complete'))
                if job['stage'] == 'train':
                    m.ab.write(Path(job['weights'])/'training_report.json', split)
                    m.ab.write(Path(job['bank'])/'fake.json', {})
                elif job['stage'] == 'eval':
                    self.assertEqual(env['OVERRIDE_INF_PROPOSALS_DIR'], str(root/'clean'))
                    m.ab.write(Path(job['metrics']), metric)
                    np.savez_compressed(attempt/'per_image_confusion.npz', ids=['v'], confusions=[matrix])
                else:
                    for stem in job['ids']:
                        folder = Path(job['cache'])/stem
                        m.ab.write(folder/'diagnostics.json', dict(seconds=1., regions=[{}]))
                        m.ab.write(folder/'complete.json', dict(result={'file': str(folder/'features.npz')}, artifacts=[]))
            def merge(_cache, _ids, raw, *_args):
                m.ab.write(raw/'fake.json', {})
            stack.enter_context(patch.object(m, 'ROOT', root))
            stack.enter_context(patch.object(m.ab, 'SCRIPT', m.ab.SCRIPT))
            stack.enter_context(patch.object(m.rpg, 'verify_records', m.rpg.verify_records))
            stack.enter_context(patch.object(m.rpg, 'baseline', return_value=(old, base, [])))
            stack.enter_context(patch.object(m.ab, 'parse_args', return_value=args))
            stack.enter_context(patch.object(m.ab, 'preflight', side_effect=lambda _: copy.deepcopy(context)))
            stack.enter_context(patch.object(m.ab, 'check_gpu'))
            stack.enter_context(patch.object(m.ab, 'source_identity', return_value=[]))
            stack.enter_context(patch.object(m, 'validate_clean_cache', return_value=dict(count=3)))
            stack.enter_context(patch.object(m.gmp, 'enumerate_members', return_value={'t1': [], 't2': []}))
            stack.enter_context(patch.object(m, 'merge', side_effect=merge))
            stack.enter_context(patch.object(m, 'compare', side_effect=lambda _a, _b, _ids, c:
                dict(smfa_miou=.85, mpsa_miou=.851, delta_pp=.1, phases=c)))
            stack.enter_context(patch.object(m.ab, 'launch', side_effect=launch))
            # Parent membership sets OVERRIDE_ variables; do not leak them to other tests.
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            cli = ['--previous', str(root/'old'), '--clean-proposals-dir', str(root/'clean'), '--version', 'test_mpsa']
            m.main(cli)
            self.assertEqual(launched[0], ('eval', 'smfa', None))
            self.assertEqual(sum(stage == 'train' for stage, _, _ in launched), 3)
            self.assertFalse(any(stage in ('pseudo', 'proposals', 'build') for stage, _, _ in launched))
            for c in (1, 2, 4):
                self.assertLess(launched.index(('train', 'mpsa', c)), launched.index(('eval', 'mpsa', c)))
            self.assertEqual(m.ab.read(root/'experiments/test_mpsa/status.json')['status'], 'complete')
            with self.assertRaises(ValueError): m.main(cli)
            count = len(launched); m.main(cli+['--resume'])
            self.assertEqual(len(launched), count)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed locally')
class TensorTests(unittest.TestCase):
    def test_reflection_shift_no_wrap(self):
        import torch
        x = torch.arange(1008, dtype=torch.float32)[None, None, None, :].expand(1, 3, 1008, 1008)
        shifted = m.shift_input(x, (7, 0))
        self.assertEqual(shifted[0, 0, 0, :10].tolist(), [7, 6, 5, 4, 3, 2, 1, 0, 1, 2])
        self.assertEqual(float(shifted[0, 0, 0, -1]), 1000)
        with self.assertRaises(ValueError): m.shift_input(x*float('nan'), (0, 0))

    def test_hard_mean_zero_finite_and_equal(self):
        import torch
        import torch.nn.functional as F
        tokens = torch.tensor([[2., 0.], [0., 1.], [10., 10.]])
        torch.testing.assert_close(m.hard_mean(tokens, [0, 1]), F.normalize(torch.tensor([1., .5]), dim=0))
        self.assertEqual(float(m.hard_mean(tokens, []).norm()), 0)
        self.assertEqual(float(m.hard_mean(tokens*0, [0]).norm()), 0)
        with self.assertRaises(ValueError): m.hard_mean(tokens*float('inf'), [0])

    def test_shared_forwards_and_single_batch_consistency(self):
        import torch
        class Model:
            resolution = 1008; context_subtract = 0; use_bf16 = False; embed_dim = 2
            calls = 0
            def _preprocess_image(self, rgb):
                return torch.zeros((1, 3, 1008, 1008)), None
            def _forward_tokens(self, inp):
                self.calls += 1
                tokens = torch.stack([torch.arange(5184).float()+1, torch.ones(5184)], dim=1)
                return tokens[None], (72, 72)
        model = Model(); stats = Counter()
        rgb = np.zeros((100, 200, 3), np.uint8)
        masks = [np.ones((100, 200), bool), np.zeros((100, 200), bool)]
        masks[1][30, 50] = True
        result = m.extract_many(model, rgb, masks, stats=stats)
        self.assertEqual(model.calls, 4); self.assertEqual(stats['image_phase_forwards'], 4)
        for count in (1, 2, 4):
            for i, mask in enumerate(masks):
                one = m.extract_many(model, rgb, [mask], (count,))[count][0]
                torch.testing.assert_close(one, result[count][i])
        self.assertEqual(len(m.extract_many(model, rgb, [], (4,))[4]), 0)


if __name__ == '__main__': unittest.main()
