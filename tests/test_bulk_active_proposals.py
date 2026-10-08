import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from run_generate_proposals_active import main
from tools.proposal_demo.test_active import fake_sam3


class BulkExportTests(unittest.TestCase):
    def test_full_export_resume_repair_and_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp, fake_sam3(), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for i in range(4):
                Image.new('RGB', (40, 32)).save(root / f'{i}.jpg')
            (root / 'model.pt').touch()
            output = root / 'out'
            args = ['--image-dir', str(root), '--checkpoint', str(root / 'model.pt'),
                    '--output-dir', str(output), '--device', 'cpu', '--precision', 'fp32',
                    '--min-mask-area', '1']
            self.assertEqual(main(args), 0)
            self.assertEqual(len(list(output.glob('*.npz'))), 4)
            for cache in output.glob('*.npz'):
                with np.load(cache) as data:
                    self.assertEqual(set(data.files), {'masks', 'scores'})
                    self.assertEqual(data['masks'].shape[1:], (32, 40))
                    self.assertEqual(data['masks'].dtype, np.uint8)
                    self.assertEqual(data['scores'].dtype, np.float32)
            with patch('run_generate_proposals_active.build_proposer', side_effect=AssertionError('should skip')):
                self.assertEqual(main(args), 0)
                manifest = output / 'export_config.json'
                old = json.loads(manifest.read_text())
                old['device'] = 'cuda:2'
                manifest.write_text(json.dumps(old))
                self.assertEqual(main(args), 0)
            (output / '0.npz').write_bytes(b'broken')
            self.assertEqual(main(args), 0)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                main(args + ['--bucket-size', '1'])

    def test_split_and_empty_proposals(self):
        with tempfile.TemporaryDirectory() as tmp, fake_sam3(), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for i in range(4):
                Image.new('RGB', (40, 32)).save(root / f'{i}.jpg')
            (root / 'model.pt').touch()
            (root / 'val.txt').write_text('1\n3\n')
            output = root / 'out'
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.device_count', return_value=1), \
                    patch('torch.cuda.set_device') as set_device:
                with self.assertRaisesRegex(ValueError, 'only 1 CUDA devices'):
                    main(['--image-dir', str(root), '--checkpoint', str(root / 'model.pt'),
                          '--output-dir', str(root / 'invalid'), '--device', 'cuda:2'])
                set_device.assert_not_called()
            main(['--image-dir', str(root), '--checkpoint', str(root / 'model.pt'),
                  '--output-dir', str(output), '--device', 'cpu', '--precision', 'fp32',
                  '--split', str(root / 'val.txt'), '--confidence-threshold', '1'])
            self.assertEqual({p.stem for p in output.glob('*.npz')}, {'1', '3'})
            with np.load(output / '1.npz') as data:
                self.assertEqual(data['masks'].shape, (0, 32, 40))
                self.assertEqual(data['scores'].shape, (0,))
