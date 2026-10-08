"""CPU-only checks for the disposable second-round launcher."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from tools import experimental_pmfa_round2 as r2


class RoundTwoTests(unittest.TestCase):
    def test_commands_are_independent_single_factor_ablations(self):
        commands = r2.make_commands(['--pseudo-dir', '/shared/labels', '--proposals-dir', '/shared/proposals'], 'round2')
        parsed = [r2.ab.parse_args(command[4:]) for _, command in commands]
        self.assertEqual([args.gpu for args in parsed], [0, 1])
        self.assertTrue(all(args.parallel_gpus is None for args in parsed))
        self.assertEqual([(args.alpha, args.steps) for args in parsed], [(.25, 2), (.5, 0)])
        self.assertEqual(parsed[0].pseudo_dir, parsed[1].pseudo_dir)
        self.assertEqual(parsed[0].proposals_dir, parsed[1].proposals_dir)
        self.assertNotEqual(parsed[0].version, parsed[1].version)

    def test_previous_inputs_resolve_reused_and_generated_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root/'labels'; labels.mkdir()
            proposals = root/'proposals-cache'; proposals.mkdir()
            manifest = dict(pmfa=asdict(r2.ab.PMFA()), precision_policy=r2.ab.POLICY, inputs=[],
                options={key:str(root/key) for key in ('dataset_root','image_dir','checkpoint',
                    'train_list','val_list','imagelevel_json','gt_dir')},
                selected=dict(pseudo=str(labels), proposals=None))
            r2.ab.write(root/'manifest.json', manifest)
            r2.ab.write(root/'comparison.json', dict(status='complete'))
            r2.ab.write(root/'proposals/complete.json', dict(result=dict(proposals=str(proposals))))
            args = r2.previous_inputs(root)
            self.assertEqual(args[-4:], ['--pseudo-dir',str(labels),'--proposals-dir',str(proposals)])
            manifest['pmfa']['alpha'] = .25
            r2.ab.write(root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'default recipe'):
                r2.previous_inputs(root)


if __name__ == '__main__':
    unittest.main()
