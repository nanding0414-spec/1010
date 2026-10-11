"""CPU tests: python -m unittest tools.proposal_demo.test_active -v

The tensor pipeline uses a deterministic SAM3 interface double, not checkpoints.
This checks batching/state/filtering contracts, not SAM3 accuracy or CUDA kernels.
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from run_sam3_proposal_demo import main, parse_args
from tools.proposal_demo.active_utils import (
    PromptClass, align_vocabulary, load_vocabulary, select_active_classes,
)
from tools.proposal_demo.active_proposer import SAM3ActiveProposer, preview_rope, skip_segmentation_head


class FakeProcessor:
    def __init__(self, model, resolution=1008, device='cpu', **kwargs):
        self.model, self.resolution = model, resolution

    def set_image(self, image):
        self.model.image_resolutions.append(self.resolution)
        return dict(original_height=image.height, original_width=image.width,
                    backbone_out=dict(resolution=self.resolution))


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_backbone = torch.nn.Module()
        self.language_backbone.resizer = torch.nn.Linear(6, 8)
        self.text_calls = 0

    def forward_text(self, words, device):
        self.text_calls += 1
        ids = torch.tensor([{'cat': 1, 'dog': 2, 'bus': 3, 'bird': 4}.get(word, 5) for word in words])
        features = ids[None, :, None].float().expand(3, len(words), 8).clone()
        embeddings = ids[None, :, None].float().expand(3, len(words), 6).clone()
        return dict(language_features=features, language_embeds=embeddings,
                    language_mask=torch.zeros(len(words), 3, dtype=torch.bool))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.image_resolutions, self.ground_calls = [], []
        self.segmentation_calls = 0

    def _get_dummy_prompt(self, num_prompts=1):
        return num_prompts

    def forward_grounding(self, backbone_out, find_input, find_target, geometric_prompt):
        count = len(find_input.text_ids)
        assert find_input.img_ids.tolist() == [0] * count
        assert find_input.text_ids.tolist() == list(range(count))
        assert geometric_prompt == count
        language = backbone_out['language_features']
        assert backbone_out['language_mask'].shape == (count, len(language))
        ids = language[0, :, 0].long()
        self.ground_calls.append((backbone_out['resolution'], count, len(language)))
        presence = torch.tensor([0.98, 0.96, 0.9, 0.01, 0.2])[ids - 1]
        out = dict(presence_logit_dec=torch.logit(presence[:, None]),
                   pred_logits=torch.full((count, 2, 1), 4.0))
        self._run_segmentation_heads(out=out, ids=ids)
        return out

    def _run_segmentation_heads(self, out, ids):
        self.segmentation_calls += 1
        masks = torch.full((len(ids), 2, 8, 8), -8.0)
        semantic = torch.full((len(ids), 1, 8, 8), -8.0)
        for k, label in enumerate(ids.tolist()):
            # Cat and dog overlap with close scores; bus has a distinct region.
            region = (slice(1, 7), slice(1, 4)) if label < 3 else (slice(1, 7), slice(5, 7))
            masks[k, :, region[0], region[1]] = 8.0
            semantic[k, :, region[0], region[1]] = 8.0
        out.update(pred_masks=masks, semantic_seg=semantic)


@contextlib.contextmanager
def fake_sam3():
    modules = {name: types.ModuleType(name) for name in (
        'sam3', 'sam3.model', 'sam3.model.sam3_image_processor',
        'sam3.model.data_misc', 'sam3.model_builder')}
    modules['sam3.model.sam3_image_processor'].Sam3Processor = FakeProcessor
    modules['sam3.model.data_misc'].FindStage = types.SimpleNamespace
    models = []

    def build(**kwargs):
        assert kwargs['enable_segmentation'] and not kwargs['enable_inst_interactivity']
        model = FakeModel()
        models.append(model)
        return model

    modules['sam3.model_builder'].build_sam3_image_model = build
    with patch.dict(sys.modules, modules), patch('src.sam3_precision.use_standard_vit_mlp', return_value=1):
        yield models


class ActiveUtilsTests(unittest.TestCase):
    def test_selection_threshold_rank_cap_and_zero_fallback(self):
        self.assertEqual(select_active_classes([.9, .8, .7, .01], .5, .25, 1, 2), [0, 1])
        self.assertEqual(select_active_classes([.1, .4, .3, .2], .9, .5, 1, 0), [1, 2])
        self.assertEqual(select_active_classes([0, 0, 0], .2, .8, 1, 0), [0])
        with self.assertRaises(ValueError):
            select_active_classes([np.nan], .2, .4, 1, 0)

    def test_dataset_names_and_ids_are_preserved(self):
        classes = align_vocabulary(['ship', 'people', 'table'])
        self.assertEqual([c.name for c in classes], ['ship', 'people', 'table'])
        self.assertEqual([c.prompt for c in classes], ['boat', 'person', 'dining table'])

    def test_vocabulary_normalization_and_alias_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vocab.json'
            path.write_text(json.dumps([dict(name='TV', prompt=' TV_monitor ', hypernyms=['display'])]))
            self.assertEqual(load_vocabulary(path)[0].prompt, 'tv monitor')
            path.write_text(json.dumps([dict(name='A', prompt='cat'), dict(name='B', prompt='CAT')]))
            with self.assertRaises(ValueError):
                load_vocabulary(path)

    def test_cli_baseline_and_active_validation(self):
        self.assertEqual(parse_args([]).backend, 'grid')
        args = parse_args(['--backend', 'active', '--no-preview'])
        self.assertFalse(args.preview)
        for flags in (['--bucket-size', '0'], ['--preview-resolution', '671'],
                      ['--background-calibration'], ['--min-active-classes', '13']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_preview_patches_restore_after_error(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_rope, self.rope_interp = True, True
                self.rope_pt_size = (72, 72)
                self.register_buffer('freqs_cis', torch.ones(72 * 72, 2, dtype=torch.complex64))

            def compute_cis(self, end_x, end_y, scale_pos):
                return torch.ones(end_x * end_y, 2, dtype=torch.complex64) * scale_pos

        model = FakeModel()
        block = torch.nn.Module()
        block.window_size = 0
        block.attn = Attention()
        model.backbone.block = block
        original = block.attn.freqs_cis
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with preview_rope(model, 672), skip_segmentation_head(model):
                self.assertEqual(block.attn.freqs_cis.shape[0], 48 * 48)
                model._run_segmentation_heads(out={}, ids=[])
                raise RuntimeError('injected')
        self.assertIs(block.attn.freqs_cis, original)
        self.assertNotIn('_run_segmentation_heads', vars(model))
        self.assertEqual(model.segmentation_calls, 0)


class ActivePipelineTests(unittest.TestCase):
    def classes(self):
        return [PromptClass(name, name, ('animal',) if name != 'bus' else ('vehicle',))
                for name in ('cat', 'dog', 'bus', 'bird')]

    def test_batched_preview_cache_keeps_close_cross_class_masks(self):
        with fake_sam3():
            model = FakeModel()
            proposer = SAM3ActiveProposer(model, torch.device('cpu'), self.classes(), bucket_size=2,
                                          min_active_classes=1, max_active_classes=3,
                                          active_keep_ratio=.25, min_mask_area=1)
            text_calls = model.backbone.text_calls
            result = proposer(np.zeros((32, 40, 3), dtype=np.uint8))
            report = proposer.last_result['report']
            self.assertEqual(report['active_class_names'], ['cat', 'dog', 'bus'])
            self.assertEqual(model.image_resolutions, [672, 1008])
            self.assertEqual(model.segmentation_calls, 2)  # Preview never runs mask head.
            self.assertEqual([c[1] for c in model.ground_calls], [2, 2, 2, 1])
            self.assertEqual([c[2] for c in model.ground_calls[:2]], [3, 3])
            self.assertTrue(all(c[2] == 6 for c in model.ground_calls[2:]))
            self.assertEqual(set(proposer.last_result['class_ids'].tolist()), {1, 2, 3})
            self.assertEqual(report['before_cap_count'], report['final_count'])
            self.assertTrue(np.logical_and(result['instances'].pred_masks[0].numpy(),
                                           result['instances'].pred_masks[1].numpy()).any())
            self.assertEqual(result['instances'].pred_masks.shape[1:], (32, 40))
            proposer(np.zeros((32, 40, 3), dtype=np.uint8))
            self.assertEqual(model.backbone.text_calls, text_calls)
            self.assertEqual(model.segmentation_calls, 4)
            self.assertNotIn('_run_segmentation_heads', vars(model))

    def test_small_vocabulary_no_context_no_preview_and_empty_masks(self):
        with fake_sam3():
            proposer = SAM3ActiveProposer(FakeModel(), torch.device('cpu'), self.classes()[:1],
                                          preview=False, context_strength=0,
                                          confidence_threshold=1.0, min_mask_area=1)
            self.assertIsNone(proposer.context)
            masks = proposer(np.zeros((10, 12, 3), dtype=np.uint8))['instances'].pred_masks
            self.assertEqual(tuple(masks.shape), (0, 10, 12))
            self.assertEqual(proposer.last_result['report']['preview_forward_calls'], 0)

    def test_bucket_size_one_matches_larger_buckets(self):
        with fake_sam3():
            outputs = []
            for size in (1, 3):
                proposer = SAM3ActiveProposer(FakeModel(), torch.device('cpu'), self.classes(),
                                              bucket_size=size, preview=False, context_strength=0,
                                              min_mask_area=1)
                instances = proposer(np.zeros((16, 16, 3), dtype=np.uint8))['instances']
                outputs.append((instances.pred_masks.numpy(), instances.scores.numpy()))
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            np.testing.assert_allclose(outputs[0][1], outputs[1][1])

    def test_three_image_cli_and_saved_diagnostics(self):
        with fake_sam3(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                Image.fromarray(np.full((32, 40, 3), 120, dtype=np.uint8)).save(root / f'{i}.png')
            checkpoint = root / 'fake.pt'
            checkpoint.touch()
            vocabulary = root / 'classes.json'
            vocabulary.write_text(json.dumps([dict(name=c.name, prompt=c.prompt, hypernyms=list(c.hypernyms))
                                              for c in self.classes()]))
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--backend', 'active', '--device', 'cpu', '--precision', 'fp32',
                      '--checkpoint', str(checkpoint), '--classes-file', str(vocabulary),
                      '--image-dir', str(root), '--image-ids', '0', '1', '2',
                      '--output-dir', str(root / 'output'), '--min-mask-area', '1'])
            runs = list((root / 'output').iterdir())
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(len(json.loads((run / 'summary.json').read_text())), 3)
            self.assertFalse((run / '3').exists())
            for i in range(3):
                folder = run / str(i)
                with np.load(folder / 'proposals.npz', allow_pickle=False) as data:
                    self.assertEqual(data['masks'].shape[1:], (32, 40))
                    self.assertEqual(len(data['masks']), len(data['class_ids']))
                self.assertEqual(Image.open(folder / 'comparison_classes.png').size, (80, 60))
                self.assertFalse((folder / 'background_mask.png').exists())
                for name in ('active_report.json', 'preview_classes.png', 'final_class_overlay.png'):
                    self.assertTrue((folder / name).is_file())

    def test_allowed_classes_and_empty_image_labels(self):
        with fake_sam3():
            model = FakeModel()
            proposer = SAM3ActiveProposer(model, torch.device('cpu'), self.classes(), min_mask_area=1,
                                          min_active_classes=1, max_active_classes=0)
            proposer(np.zeros((16, 16, 3), dtype=np.uint8), class_indices=[1, 2])
            self.assertEqual(set(proposer.last_result['class_ids'].tolist()), {2, 3})
            self.assertIsNone(proposer.last_result['report']['preview_presence'][0])
            before = len(model.image_resolutions)
            result = proposer(np.zeros((16, 16, 3), dtype=np.uint8), class_indices=[])['instances']
            self.assertEqual(len(result.scores), 0)
            self.assertEqual(len(model.image_resolutions), before)

    def test_before_after_colors_follow_class_ids(self):
        from tools.proposal_demo.visualize import class_overlay
        rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        masks = np.zeros((2, 64, 64), dtype=bool)
        masks[0, 4:12, 4:12] = True
        masks[1, 8:56, 8:56] = True
        classes = [dict(name=name) for name in ('cat', 'dog', 'bus')]
        before = class_overlay(rgb, masks, np.array([.9, .8]), np.array([1, 3]), classes)
        after = class_overlay(rgb, masks[1:], np.array([.8]), np.array([3]), classes)
        np.testing.assert_array_equal(before[45, 45], after[45, 45])


if __name__ == '__main__':
    unittest.main()
