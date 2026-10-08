import tempfile
import unittest
from pathlib import Path
from tools import experimental_background_partition_recovery as recovery


class ArtifactRecoveryTests(unittest.TestCase):
    def test_order_and_identical_duplicates_preserve_strict_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.write_bytes(b'foreground'); b.write_bytes(b'background')
            records = recovery.ab.identities([a,b])
            recovery.verified_artifacts([records[1], records[0], records[0]])
            a.write_bytes(b'Foreground')
            with self.assertRaises(ValueError):
                recovery.verified_artifacts(records)

    def test_conflicting_duplicate_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'file'; file.write_bytes(b'model')
            records = recovery.ab.identities([file])
            with self.assertRaises(ValueError):
                recovery.verified_artifacts(records+[dict(records[0], sha256='different')])
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                recovery.verified_artifacts(records)

    def test_resume_does_not_modify_marker_or_reexecute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); artifact=root/'raw'; artifact.write_bytes(b'raw')
            record=recovery.ab.identities([artifact])[0]
            marker=root/'experiment/merge/complete.json'
            recovery.ab.write(marker, dict(result=dict(raw='old'), artifacts=[record,record],seconds=4))
            before=marker.read_bytes()
            stages=recovery.AliasAwareStages(root/'experiment',True)
            stages.verify_all()
            self.assertEqual(stages.run('merge',lambda _:self.fail('must reuse')),dict(raw='old'))
            self.assertEqual(marker.read_bytes(),before)

    def test_new_attempt_still_uses_original_stage_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages=recovery.AliasAwareStages(Path(tmp),True)
            def action(attempt):
                file=attempt/'result';file.write_bytes(b'new')
                return dict(ok=True),[file]
            self.assertEqual(stages.run('eval_safe_matched_42',action),dict(ok=True))
            stages.verify_all()


if __name__=='__main__':unittest.main()
