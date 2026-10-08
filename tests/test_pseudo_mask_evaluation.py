import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.pseudo_mask_evaluation import (
    accumulate_confusion, evaluate_pseudo_masks, find_ground_truth, gt_to_internal,
    summarize_confusion,
)


class PseudoEvaluationTests(unittest.TestCase):
    def test_known_iou_and_ignore_pixels(self):
        # Class 0: TP=1, union=2. Class 1: TP=1, union=2. Last pixel ignored.
        confusion = np.zeros((2, 2), dtype=np.int64)
        count = accumulate_confusion(confusion, np.array([[0, 0, 1, 0]]), np.array([[0, 1, 1, 255]]))
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(confusion, [[1, 0], [1, 1]])
        report = summarize_confusion(confusion, ['cat'], 'voc')
        self.assertAlmostEqual(report['mIoU'], .5)
        self.assertAlmostEqual(report['foreground_mIoU'], .5)

    def test_global_counts_and_predicted_only_classes(self):
        confusion = np.zeros((4, 4), dtype=np.int64)
        accumulate_confusion(confusion, np.array([[1]]), np.array([[1]]))
        accumulate_confusion(confusion, np.full((1, 9), 2), np.full((1, 9), 1))
        report = summarize_confusion(confusion, ['cat', 'dog', 'bird'], 'voc')
        self.assertAlmostEqual(report['mIoU'], .05)  # cat 1/10, predicted-only dog 0.
        self.assertIsNone(report['per_class'][3]['IoU'])
        self.assertEqual(report['per_class'][2]['IoU'], 0)

    def test_ade_and_city_background_prediction_counts_as_error(self):
        for dataset, gt, path in [('ade20k', np.array([[0, 1, 2, 255]]), 'a.png'),
                                   ('cityscapes', np.array([[0, 7, 8, 255]]), 'a_labelIds.png'),
                                   ('cityscapes', np.array([[255, 0, 1, 255]]), 'a_labelTrainIds.png')]:
            with self.subTest(dataset=dataset, path=path):
                normalized = gt_to_internal(gt, dataset, path)
                np.testing.assert_array_equal(normalized, [[255, 1, 2, 255]])
                confusion = np.zeros((3, 3), dtype=np.int64)
                accumulate_confusion(confusion, np.array([[0, 0, 2, 0]]), normalized)
                report = summarize_confusion(confusion, ['a', 'b'], dataset)
                self.assertAlmostEqual(report['mIoU'], .5)
                self.assertEqual(report['per_class'][1]['IoU'], 0)
                self.assertFalse(report['includes_background'])

    def test_invalid_shapes_and_labels_fail_without_mutating_counts(self):
        confusion = np.zeros((2, 2), dtype=np.int64)
        for pred, gt in [(np.zeros((2, 2), dtype=int), np.zeros((1, 1), dtype=int)),
                         (np.array([[255]]), np.array([[1]])),
                         (np.array([[0]]), np.array([[3]]))]:
            with self.assertRaises(ValueError):
                accumulate_confusion(confusion, pred, gt)
        self.assertEqual(confusion.sum(), 0)

    def test_gt_lookup_and_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / 'SegmentationClassAug'
            gt_dir.mkdir()
            gt = np.array([[0, 1], [1, 255]], dtype=np.uint8)
            Image.fromarray(gt).save(gt_dir / 'a.png')
            pred = root / 'prediction.png'
            Image.fromarray(np.array([[0, 1], [0, 0]], dtype=np.uint8)).save(pred)
            image = root / 'JPEGImages/a.jpg'
            self.assertEqual(find_ground_truth(image, root, 'voc', 'trainaug')[0], gt_dir / 'a.png')
            samples = [dict(image=image, prediction=pred),
                       dict(image=root / 'b.jpg', prediction=pred),
                       dict(image=root / 'c.jpg', prediction=pred, generation_failed=True)]
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluate_pseudo_masks(samples, root, 'voc', 'trainaug', ['cat'], root / 'report.json')
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['evaluated_count'], 1)
            self.assertEqual(len(report['missing_gt']), 1)
            self.assertEqual(len(report['missing_predictions']), 1)
            self.assertAlmostEqual(report['mIoU'], .5)
            self.assertNotIn('NaN', (root / 'report.json').read_text())
            gt = np.full((2, 2), 255, dtype=np.uint8)
            Image.fromarray(gt).save(gt_dir / 'a.png')
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluate_pseudo_masks(samples[:1], root, 'voc', 'trainaug', ['cat'], root / 'report.json')
            self.assertEqual(report['status'], 'unavailable')
            self.assertIsNone(report['mIoU'])


class ActivePseudoIntegrationTests(unittest.TestCase):
    def test_x2_images_saved_at_original_size_and_evaluated(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import main
        with fake_sam3(), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_dataset(root)
            enlarged = root / 'JPEGImages_SPAN_x2'
            enlarged.mkdir()
            for i in range(2):
                Image.new('RGB', (32, 32)).save(enlarged / f'{i}.png')
            checkpoint = root / 'fake.pt'
            checkpoint.touch()
            args = ['--split', 'train', '--min-mask-area', '0', '--dataset-root', str(root), '--image-dir', str(enlarged),
                    '--output-size-reference-dir', str(root / 'JPEGImages'),
                    '--output-dir', str(root / 'pseudo'), '--experiment-dir', str(root / 'experiment'),
                    '--device', 'cpu', '--no-bf16', '--ckpt', str(checkpoint)]
            self.assertEqual(main(args), 0)
            with Image.open(root / 'pseudo/0.png') as pred:
                self.assertEqual(pred.size, (16, 16))
                self.assertTrue(set(np.unique(pred)).issubset({0, 6, 8, 12}))
            report = json.loads((root / 'experiment/miou_train.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['evaluated_count'], 2)
            self.assertIn('JPEGImages_SPAN_x2', report['evaluated'][0]['image'])

    def make_dataset(self, root):
        (root / 'JPEGImages').mkdir()
        (root / 'ImageSets/ImageLevel').mkdir(parents=True)
        (root / 'SegmentationClass').mkdir()
        records = []
        for i, labels in enumerate(([8, 12, 6], [])):
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / f'JPEGImages/{i}.jpg')
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(root / f'SegmentationClass/{i}.png')
            # IDs intentionally differ from file stems: label lookup must use file_name.
            records.append(dict(img_id=100 + i, file_name=f'JPEGImages/{i}.jpg', labels=labels))
        (root / 'ImageSets/ImageLevel/train_imagelevel.json').write_text(json.dumps(dict(images=records)))

    def test_active_generation_preserves_ids_overlaps_and_evaluates_saved_pngs(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator, main
        with fake_sam3() as models, tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_dataset(root)
            generator = SAM3PseudoMaskGenerator(
                output_dir=root/'pseudo', dataset_root=root, ckpt='fake.pt', device='cpu', use_bf16=False,
                prompt_backend='active', min_mask_area=1,
                active_options=dict(min_active_classes=1, max_active_classes=0, bucket_size=2))
            result = generator.generate_all()
            self.assertFalse(result['errors'])
            self.assertEqual(result['evaluation']['evaluated_count'], 2)
            self.assertEqual(result['evaluation']['valid_pixels'], 512)
            with Image.open(generator.output_dir / '0.png') as image:
                pred = np.array(image)
            self.assertEqual(set(np.unique(pred)), {0, 6, 8})  # cat wins cat/dog overlap; bus retains ID 6.
            self.assertEqual(pred[5, 4], 8)
            with Image.open(generator.output_dir / '1.png') as image:
                self.assertEqual(np.array(image).sum(), 0)
            report = json.loads((generator.output_dir / 'meta/prompt_choices/train/0.json').read_text())
            self.assertEqual(report['candidate_class_ids'], [8, 12, 6])
            calls = len(models[0].ground_calls)
            resumed = generator.generate_all()
            self.assertEqual(resumed['resumed'], 2)
            self.assertEqual(len(models[0].ground_calls), calls)
            self.assertEqual(main(['--split', 'train', '--min-mask-area', '0', '--dataset-root', str(root), '--output-dir', str(generator.output_dir),
                                   '--eval-only', '--device', 'cpu']), 0)
            self.assertEqual(len(models), 1)  # eval-only never builds SAM3.
            generator.active_options['bucket_size'] = 1
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                generator.generate_all()

    def test_failed_regeneration_cannot_score_a_stale_prediction(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
        with fake_sam3(), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_dataset(root)
            generator = SAM3PseudoMaskGenerator(output_dir=root/'pseudo', dataset_root=root, ckpt='fake.pt', device='cpu', use_bf16=False,
                                                 prompt_backend='active', min_mask_area=1)
            generator.generate_all()
            def fail(*args):
                raise ValueError('injected failure')
            generator.generate_mask = fail
            result = generator.generate_all(resume=False)
            self.assertEqual(len(result['errors']), 2)
            self.assertEqual(result['evaluation']['evaluated_count'], 0)
            self.assertIsNone(result['evaluation']['mIoU'])
            retry = generator.generate_all()
            self.assertEqual(retry['resumed'], 0)
            self.assertEqual(len(retry['errors']), 2)

    def test_partial_no_resume_does_not_adopt_old_settings_for_remaining_images(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
        with fake_sam3(), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_dataset(root)
            generator = SAM3PseudoMaskGenerator(output_dir=root/'pseudo', dataset_root=root, ckpt='fake.pt', device='cpu', use_bf16=False,
                                                 prompt_backend='active', min_mask_area=1)
            generator.generate_all()
            generator.min_mask_area = 2
            generator.active_proposer.min_mask_area = 2
            generator.generate_all(resume=False, limit=1)
            completed = generator.generate_all()
            self.assertEqual(completed['resumed'], 1)
            self.assertEqual(completed['success'], 2)

    def test_cli_defaults_and_removed_prompt_route_rejected(self):
        from tools.generate_pseudo_masks_sam3 import parse_args
        self.assertEqual(parse_args([]).prompt_backend, 'active')
        self.assertTrue(parse_args([]).evaluate)
        with self.assertRaises(SystemExit):
            parse_args(['--use-synonyms'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--prompt-backend', 'active', '--use-synonyms'])

    def test_missing_manifest_fails_before_model_load(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import main
        with fake_sam3() as models, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'imagelevel-json'):
                main(['--dataset-root', tmp, '--split', 'trainaug', '--device', 'cpu'])
            self.assertEqual(len(models), 0)

    def test_external_manifest_preflight_is_reused(self):
        from tools.proposal_demo.test_active import fake_sam3
        from tools.generate_pseudo_masks_sam3 import main
        with fake_sam3() as models, tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.make_dataset(root)
            manifest = root / 'external_labels.json'
            (root / 'ImageSets/ImageLevel/train_imagelevel.json').replace(manifest)
            result = main(['--split', 'train', '--min-mask-area', '0', '--output-dir', str(root/'pseudo/sam3_active'), '--dataset-root', str(root), '--imagelevel-json', str(manifest),
                           '--device', 'cpu', '--no-bf16', '--ckpt', 'fake.pt', '--min-mask-area', '1'])
            self.assertEqual(result, 0)
            self.assertEqual(len(models), 1)
            report = json.loads((root / 'pseudo/sam3_active/meta/miou_train.json').read_text())
            self.assertEqual(report['evaluated_count'], 2)


if __name__ == '__main__':
    unittest.main()
