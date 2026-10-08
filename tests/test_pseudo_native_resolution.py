import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from tools.pseudo_mask_evaluation import evaluate_pseudo_masks


class NativeResolutionTests(unittest.TestCase):
    def test_alignment_leaves_saved_png_unchanged_and_defaults_strict(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / 'SegmentationClass').mkdir()
            gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            Image.fromarray(gt).save(root / 'SegmentationClass/a.png')
            Image.new('RGB', (4, 4)).save(root / 'a.jpg')
            Image.fromarray(gt.repeat(2, 0).repeat(2, 1)).save(root / 'pred.png')
            original = (root / 'pred.png').read_bytes()
            samples = [dict(image=root / 'a.jpg', prediction=root / 'pred.png')]
            args = (samples, root, 'voc', 'train', ['object'], root / 'metric.json')
            self.assertEqual(evaluate_pseudo_masks(*args)['status'], 'unavailable')
            result = evaluate_pseudo_masks(*args, resize_prediction_to_gt=True)
            self.assertEqual(result['mIoU'], 1.)
            self.assertEqual(len(result['resized_predictions']), 1)
            self.assertEqual((root / 'pred.png').read_bytes(), original)
            bad = gt.repeat(2, 0).repeat(2, 1)
            bad[0, 0] = 255
            Image.fromarray(bad).save(root / 'pred.png')
            self.assertEqual(evaluate_pseudo_masks(*args, resize_prediction_to_gt=True)['status'], 'unavailable')

    def test_generation_and_evaluation_preserve_x2_output(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import main
        from tests.test_pseudo_mask_evaluation import ActivePseudoIntegrationTests
        with fake_sam3(), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            ActivePseudoIntegrationTests().make_dataset(root)
            enlarged = root / 'x2'
            enlarged.mkdir()
            for i in range(2):
                Image.new('RGB', (32, 32)).save(enlarged / f'{i}.png')
            self.assertEqual(main(['--split', 'train', '--min-mask-area', '0', '--dataset-root', str(root), '--image-dir', str(enlarged),
                '--output-dir', str(root / 'out'), '--device', 'cpu', '--no-bf16',
                '--ckpt', 'fake.pt', '--eval-resize-prediction-to-gt']), 0)
            with Image.open(root / 'out/0.png') as image:
                self.assertEqual(image.size, (32, 32))
            report = json.loads((root / 'out/meta/miou_train.json').read_text())
            self.assertEqual(report['evaluated_count'], 2)
            self.assertEqual(report['prediction_alignment'], 'nearest_to_gt_in_memory')
