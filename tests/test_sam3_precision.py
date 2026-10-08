import ast
from functools import partial
from pathlib import Path
import unittest

import torch
from torch import nn
from src.sam3_precision import use_standard_vit_mlp


class PrecisionTests(unittest.TestCase):
    def test_actual_bundled_mlp_fp32_failure_and_fallback(self):
        root = Path(__file__).resolve().parents[1] / 'sam3-main/sam3'
        fused = {}
        exec(compile((root / 'perflib/fused.py').read_text(), 'fused.py', 'exec'), fused)
        tree = ast.parse((root / 'model/vitdet.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mlp')
        namespace = dict(__name__='sam3.model.vitdet', nn=nn, torch=torch, partial=partial,
                         addmm_act=fused['addmm_act'])
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'vitdet.Mlp', 'exec'), namespace)
        model = namespace['Mlp'](4, hidden_features=8).eval()
        x = torch.randn(2, 3, 4)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'same dtype'):
                model(x)
            expected = model.fc2(model.norm(model.act(model.fc1(x))))
            self.assertEqual(use_standard_vit_mlp(model), 1)
            actual = model(x)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, weights[name])

    def test_runtime_error_stops_generation(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import Mock
        from tools.proposal_demo.test_active import fake_sam3
        from tests.test_pseudo_mask_evaluation import ActivePseudoIntegrationTests
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
        with tempfile.TemporaryDirectory() as tmp, fake_sam3(), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            ActivePseudoIntegrationTests().make_dataset(root)
            generator = SAM3PseudoMaskGenerator(output_dir=root/'pseudo', dataset_root=root, ckpt='fake.pt', device='cpu',
                use_bf16=False, prompt_backend='active')
            generator.generate_mask = Mock(side_effect=RuntimeError('dtype mismatch'))
            with self.assertRaisesRegex(RuntimeError, 'Generation stopped'):
                generator.generate_all()
            self.assertEqual(generator.generate_mask.call_count, 1)
            record = json.loads((generator.output_dir / 'meta/run_train.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertFalse((generator.output_dir / 'meta/miou_train.json').exists())
