import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.run_pseudo_two_gpus import main, partition_records


class TwoGpuTests(unittest.TestCase):
    def test_partition_no_overlap_and_full_coverage(self):
        rows = [dict(file_name=f'{i}.jpg') for i in range(7)]
        shards = partition_records(rows)
        self.assertEqual([len(s) for s in shards], [4, 3])
        ids = [{r['file_name'] for r in s} for s in shards]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(len(ids[0] | ids[1]), 7)
        with self.assertRaises(ValueError):
            partition_records([dict(file_name='a/0.jpg'), dict(file_name='b/0.png')])

    def test_workers_use_separate_metadata_and_full_evaluation(self):
        from tools.generate_pseudo_masks_sam3 import main as generate
        from tools.proposal_demo.test_active import fake_sam3
        from tests.test_pseudo_mask_evaluation import ActivePseudoIntegrationTests
        calls = []
        class LocalProcess:
            def __init__(self, command, **kwargs):
                calls.append(command)
                self.pid = len(calls)
                self.returncode = generate(command[3:] + ['--device', 'cpu'])
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                return self.returncode
        with tempfile.TemporaryDirectory() as tmp, fake_sam3(), contextlib.redirect_stdout(io.StringIO()), \
                patch('torch.cuda.device_count', return_value=2), \
                patch('torch.cuda.get_device_name', return_value='fake GPU'), \
                patch('tools.run_pseudo_two_gpus.subprocess.Popen', LocalProcess):
            root = Path(tmp)
            ActivePseudoIntegrationTests().make_dataset(root)
            (root / 'model.pt').touch()
            args = ['--split', 'train', '--min-mask-area', '0', '--dataset-root', str(root), '--imagelevel-json', str(root / 'ImageSets/ImageLevel/train_imagelevel.json'),
                    '--output-dir', str(root / 'pseudo'), '--experiment-dir', str(root / 'experiment'),
                    '--ckpt', str(root / 'model.pt'), '--no-bf16']
            self.assertEqual(main(args), 0)
            self.assertEqual(len(calls), 3)
            self.assertIn('cuda:0', calls[0])
            self.assertIn('cuda:1', calls[1])
            for i in range(2):
                self.assertTrue((root / f'pseudo/{i}.png').is_file())
                self.assertTrue((root / f'pseudo/meta/shard_{i}/completed_train.json').is_file())
            report = json.loads((root / 'experiment/evaluation/miou_train.json').read_text())
            self.assertEqual(report['evaluated_count'], 2)
            self.assertEqual(report['status'], 'complete')
            with self.assertRaises(FileExistsError):
                main(args)
