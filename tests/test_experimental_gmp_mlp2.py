import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from tools import experimental_gmp_mlp2 as run
from tests.test_experimental_smfa_mlp_queue import metrics


class UnitTests(unittest.TestCase):
    def test_depth_change_preserves_seed_and_every_other_argument(self):
        args = ['trainer', '--num-layers', '3', '--seed', '2026', '--split-file', 'fixed', '--hidden-dim', '512']
        expected = args.copy()
        expected[2] = '2'
        self.assertEqual(run.depth_two_argv(args), expected)
        self.assertEqual(args[2], '3')
        with self.assertRaises(ValueError):
            run.depth_two_argv(['--num-layers', '3'])

    def test_wrong_group_signature_or_baseline_training_refused(self):
        sig = run.queue.signature({}, 'combo', run.queue.Recipe('C', 'a'*64))
        job = dict(stage='train', role='gmp3', seed=42, group='C', raw_signature=sig, checkpoint_sha256='a'*64)
        with self.assertRaises(ValueError):
            run.worker(job)
        for group in ('D', 'SMFA'):
            bad = dict(job, role='gmp2', group=group)
            with self.assertRaises(ValueError):
                run.worker(bad)
        for field, value in [('gmp_rho', .2), ('phases', 2)]:
            bad = copy.deepcopy(job)
            bad['role'] = 'gmp2'
            bad['raw_signature']['experimental_pooling'][field] = value
            with self.assertRaises(ValueError):
                run.worker(bad)

    def test_worker_routes_gmp2_without_changing_pooling(self):
        sig = run.queue.signature({}, 'combo', run.queue.Recipe('C', 'a'*64))
        job = dict(stage='eval', role='gmp2', seed=42, group='C', raw_signature=sig, checkpoint_sha256='a'*64)
        with patch.object(run.mlp, 'install_adapter') as install, patch.object(run.queue, 'worker') as worker:
            run.worker(job)
            install.assert_called_once_with('d2')
            worker.assert_called_once_with(job)
            self.assertEqual(job['group'], 'C')

    def test_training_wrappers_preserve_real_seed_and_depth(self):
        sig = run.queue.signature({}, 'combo', run.queue.Recipe('C', 'a'*64))
        with tempfile.TemporaryDirectory() as tmp:
            job = dict(stage='train', role='gmp2', seed=123, group='C', raw_signature=sig,
                       checkpoint_sha256='a'*64, attempt=tmp)
            def combo_worker(job):
                # The existing combo wrapper substitutes seed before calling our wrapper.
                run.sys.argv = ['trainer', '--seed', '123', '--num-layers', '3', '--split-file', 'fixed']
                return run.ab.runpy.run_path(str(run.ROOT / 'tools/train_feature_adapter.py'), run_name='__main__')
            with patch.object(run.mlp, 'install_adapter'), patch.object(run.queue, 'worker', side_effect=combo_worker), \
                 patch.object(run.ab.runpy, 'run_path') as original, patch.object(run.sys, 'argv', []):
                run.worker(job)
                original.assert_called_once()
                effective = run.ab.read(Path(tmp) / 'effective_training_argv.json')
                self.assertEqual(effective['argv'][2], '123')
                self.assertEqual(effective['argv'][4], '2')

    def test_partial_reports_do_not_claim_record_and_interaction(self):
        report = dict(miou=dict(smfa3=.85, smfa2=.851, gmp3=.852, gmp2=.854),
                      delta_pp={'gmp2-smfa2': .3, 'gmp2-gmp3': .2, 'gmp2-smfa3': .4, 'interaction': .1})
        partial = run.summarize({'42': report})
        self.assertIsNone(partial['new_high_mean_vs_references'])
        full = run.summarize({str(s): report for s in run.SEEDS})
        self.assertTrue(full['new_high_mean_vs_references'])
        self.assertAlmostEqual(full['contrasts']['interaction']['mean_pp'], .1)


class SimulationTests(unittest.TestCase):
    def case(self, root, resume=False, fail_replay=False):
        split = dict(train_images=['train'], validation_images=['heldout'])
        refs = {role: {} for role in ('smfa3', 'smfa2', 'gmp3')}
        raw = root / 'raw'
        raw.mkdir(exist_ok=True)
        signature = {'format_version': 3006}
        for role in refs:
            for seed in run.SEEDS:
                w, b = root / f'{role}_{seed}_weights', root / f'{role}_{seed}_bank'
                run.ab.write(w / 'training_report.json', split)
                run.ab.write(b / 'meta/feature_signature.json', signature)
                metric = root / f'{role}_{seed}_metrics.json'
                run.ab.write(metric, metrics(signature))
                refs[role][seed] = dict(raw=str(raw), weights=str(w), bank=str(b), metrics=str(metric), per_image='old.npz')
        common = {k: str(root / k) for k in ('dataset_root', 'image_dir', 'checkpoint', 'train_list', 'val_list',
            'imagelevel_json', 'gt_dir', 'cache')}
        common.update(images={}, train_ids=['train'], val_ids=metrics()['evaluated_ids'], cache_identity={},
            checkpoint_sha256='a'*64, raw_signature=signature, proposals='fixed-clean')
        calls = []
        def launch(job, attempt, env, log, cancel, deadline):
            calls.append((job['role'], job['seed'], job['stage']))
            if fail_replay:
                raise run.queue.GlobalFailure('Historical matrix mismatch')
            run.ab.write(attempt / 'queue_worker.json', dict(status='complete'))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('simulation', encoding='utf-8')
            if job['stage'] == 'train':
                self.assertEqual(job['group'], 'C')
                run.ab.write(Path(job['weights']) / 'training_report.json', split)
                run.ab.write(attempt / 'effective_training_argv.json', dict(seed=job['seed'], architecture=run.mlp.adapter_tag('d2')))
                run.ab.write(Path(job['bank']) / 'meta/feature_signature.json', dict(adapter=dict(
                    architecture=dict(experimental_mlp=run.mlp.adapter_tag('d2')))))
            else:
                run.ab.write(Path(job['metrics']), metrics(run.ab.read(Path(job['bank']) / 'meta/feature_signature.json')))
                (attempt / 'per_image_confusion.npz').write_bytes(b'simulation')
        with patch.object(run, 'ROOT', root), patch.object(run.os, 'chdir'), \
             patch.object(run, 'history', return_value=(common, refs, split, [], {'count': 3609}, str(raw), signature)), \
             patch.object(run.ab, 'check_gpu'), patch.object(run.ab, 'source_identity', return_value=[]), \
             patch.object(run.ab, 'bank_members', return_value={'class': {'region'}}), \
             patch.object(run.mlp.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*2**30)), \
             patch.object(run.mlp, 'launch', side_effect=launch), patch.object(run.seeds, 'validate_image_matrices'):
            args = ['--version', 'gmp_test'] + (['--resume'] if resume else [])
            if fail_replay:
                with self.assertRaises(run.queue.GlobalFailure):
                    run.main(args)
            else:
                run.main(args)
        return calls, run.ab.read(root / 'experiments/gmp_test/summary.json')

    def test_six_replays_then_three_trainings_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, report = self.case(root)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(calls), 12)
            self.assertTrue(all(stage == 'eval' and role != 'gmp2' for role, seed, stage in calls[:6]))
            self.assertEqual(sum(stage == 'train' for role, seed, stage in calls), 3)
            self.assertEqual(len(report['results']), 3)
            calls, report = self.case(root, resume=True)
            self.assertEqual(calls, [])
            with self.assertRaises(ValueError):
                self.case(root)

    def test_failed_history_never_starts_new_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, report = self.case(Path(tmp), fail_replay=True)
            self.assertEqual(report['status'], 'incomplete')
            self.assertFalse(any(stage == 'train' for role, seed, stage in calls))


if __name__ == '__main__':
    unittest.main()
