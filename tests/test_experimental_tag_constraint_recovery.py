import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import experimental_tag_constraint_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def test_evaluation_paths_override_historical_environment_without_mutation(self):
        env = dict(OVERRIDE_INF_FEATURE_BANK_ROOT='old-bank', CUDA_VISIBLE_DEVICES='1',
            OVERRIDE_FEATURE_ADAPTER_CKPT='old-adapter', OVERRIDE_INF_PROPOSALS_DIR='old-masks')
        for seed in (42, 123, 2026):
            job = dict(stage='eval', weights=f'new/{seed}', bank=f'bank/{seed}', proposals='clean')
            bound = recovery.evaluation_environment(job, env)
            self.assertEqual(bound['OVERRIDE_INF_FEATURE_BANK_ROOT'], job['bank'])
            self.assertEqual(Path(bound['OVERRIDE_FEATURE_ADAPTER_CKPT']), Path(job['weights'])/'adapter.pt')
            self.assertEqual(bound['OVERRIDE_INF_PROPOSALS_DIR'], 'clean')
            self.assertEqual(bound['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['OVERRIDE_INF_FEATURE_BANK_ROOT'], 'old-bank')
        self.assertEqual(recovery.evaluation_environment(dict(stage='train'), env), env)

    def test_completed_training_reused_read_only_and_verified_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); artifact = root/'adapter.pt'; artifact.write_bytes(b'model')
            old = root/'previous'; output = root/'new'
            value = dict(result=dict(weights='weights', bank='bank', attempt='attempt'),
                artifacts=recovery.ab.identities([artifact]), seconds=5.)
            cls = recovery.stages_with_recovery(recovery.ab.Stages, old, {'train_control_42':value})
            stages = cls(output, True)
            with patch.object(recovery.gmp, 'verify', wraps=recovery.gmp.verify) as check:
                result = stages.run('train_control_42', lambda _: self.fail('must not train again'))
                self.assertEqual(result, value['result']); check.assert_called_once()
            self.assertFalse(old.exists())
            stages.verify_all()
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):stages.run('train_control_42', lambda _: None)

    def test_incomplete_training_executes_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cls = recovery.stages_with_recovery(recovery.ab.Stages, Path(tmp)/'old', {})
            stages = cls(Path(tmp)/'new', True)
            def action(attempt):
                file = attempt/'result.json'; recovery.ab.write(file, dict(ok=True))
                return dict(attempt=str(attempt)), [file]
            result = stages.run('train_control_2026', action)
            self.assertTrue(Path(result['attempt']).exists()); stages.verify_all()


if __name__ == '__main__':unittest.main()
