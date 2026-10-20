"""Small CPU contracts for R supplementary proposals; no weights or GT required."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import oracle_guided_r as r


class GeometryAndPolicyTests(unittest.TestCase):
    def test_alias_replaces_only_main_slot_not_canonical_context(self):
        text = dict(language_features=np.arange(3 * 20 * 4, dtype=np.float32).reshape(3, 20, 4),
                    language_mask=np.zeros((20, 3), bool))
        context = np.arange(4 * 20 * 4, dtype=np.float32).reshape(4, 20, 4)
        before = r.tensor_hash(context)
        replacement = dict(language_features=np.full((3, 1, 4), -1., np.float32),
                           language_mask=np.asarray([[False, True, True]]))
        new = r.replace_main_prompt(text, replacement, 10)
        other = [i for i in range(20) if i != 10]
        np.testing.assert_array_equal(new['language_features'][:, other], text['language_features'][:, other])
        np.testing.assert_array_equal(new['language_mask'][other], text['language_mask'][other])
        np.testing.assert_array_equal(new['language_features'][:, 10:11], replacement['language_features'])
        self.assertEqual(r.tensor_hash(context), before)
        self.assertFalse((text['language_features'] < 0).any())
        broken = dict(replacement, language_features=np.ones((2, 1, 4)))
        with self.assertRaises(ValueError):
            r.replace_main_prompt(text, broken, 10)

    def test_odd_image_crop_maps_exact_pixels_and_records_boundary_contact(self):
        box = r.center_crop_box(101, 63)
        self.assertEqual(box, (6, 10, 56, 90))
        crop = np.zeros((2, 80, 50), bool)
        crop[0, 4:6, 8:10] = True
        crop[1, :2, 8:10] = True
        full, touches = r.map_crop_masks(crop, box, (101, 63))
        np.testing.assert_array_equal(full[:, 10:90, 6:56], crop)
        self.assertEqual(int(full.sum()), int(crop.sum()))
        self.assertEqual(touches.tolist(), [False, True])
        self.assertFalse(full[:, :10].any())

    def test_hash_dedup_precedes_per_source_class_cap_with_stable_ties(self):
        a = np.zeros((30, 30), bool); a[1:12, 1:12] = True
        b = np.zeros_like(a); b[15:27, 1:12] = True
        c = np.zeros_like(a); c[15:27, 15:27] = True
        masks = np.stack([a, a, b, c])
        scores = [.9, .9, .85, .8]
        classes = [2, 1, 1, 1]
        identifiers = ['b', 'a', 'c', 'd']
        result, report = r.filter_and_cap(masks, scores, classes, identifiers)
        self.assertEqual(result['candidate_ids'].tolist(), ['a'])
        self.assertEqual(result['candidate_ids_pre_cap'].tolist(), ['a', 'c', 'd'])
        self.assertEqual(report['after_hash_dedup'], 3)
        self.assertEqual(report['after_class_cap'], 1)
        permutation = np.asarray([3, 1, 0, 2])
        again, _ = r.filter_and_cap(masks[permutation], np.asarray(scores)[permutation],
                                    np.asarray(classes)[permutation], np.asarray(identifiers)[permutation])
        np.testing.assert_array_equal(result['candidate_ids'], again['candidate_ids'])
        np.testing.assert_array_equal(result['masks'], again['masks'])

    def test_crop_area_is_screened_against_original_image_not_crop(self):
        crop = np.ones((1, 8, 8), bool)
        full, _ = r.map_crop_masks(crop, (1, 1, 9, 9), (10, 10))
        data, _ = r.filter_and_cap(full, [.9], [11], ['r3'], min_area=1, max_area_ratio=.95)
        self.assertEqual(len(data['masks']), 1)
        data, _ = r.filter_and_cap(crop, [.9], [11], ['r3'], min_area=1, max_area_ratio=.95)
        self.assertEqual(len(data['masks']), 0)

    def test_strict_addition_uses_retrieved_class_and_preserves_existing_foreground(self):
        base = np.zeros((5, 5), np.uint8); base[0, :] = 8
        masks = np.ones((2, 5, 5), bool)
        query = dict(class_ids=np.asarray([5, 11]), source_classes=np.asarray([11, 5]),
                     proposal_scores=np.asarray([.9, .9]), retrieval_scores=np.asarray([.98, .99]),
                     margins=np.asarray([.2, .09]), fused_scores=np.asarray([.9, .99]),
                     candidate_ids=np.asarray(['qualifies', 'fails_margin']))
        out, report = r.strict_background_rescue(base, masks, query)
        np.testing.assert_array_equal(out[0], base[0])
        self.assertTrue((out[1:] == 5).all())
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(report['accepted_candidates'][0]['source_class'], 11)
        self.assertEqual(report['accepted_candidates'][0]['class_id'], 5)
        self.assertEqual(report['added_pixels'], 20)

    def test_checkpoint_reuse_rejects_changed_artifact_and_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'image'; folder.mkdir()
            path = folder / 'prediction.png'; path.write_bytes(b'committed')
            binding = dict(signature='frozen')
            r._complete(folder, binding, [path], {})
            self.assertIsNotNone(r._existing(folder, binding))
            with self.assertRaises(ValueError):
                r._existing(folder, dict(signature='changed'))
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                r._existing(folder, binding)

    def test_token_receipt_geometry_rejects_crop_tokens_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / 'tokens.json'
            receipt.write_text(json.dumps(dict(binding=dict(encoder_signature={'embed_dim': 1280},
                original_shape=[80, 80]), sha256='not-consumed')), encoding='utf-8')
            manifest = dict(source_signature=dict(embed_dim=1280), token_records={'id':
                dict(path=str(root/'tokens.npz'), sha256='not-consumed', receipt_path=str(receipt),
                     receipt_sha256=r.sha256(receipt), geometry=dict(original_hw=[100, 100], token_hw=[72, 72]))})
            with self.assertRaisesRegex(ValueError, 'Original token receipt'):
                r.load_original_tokens(manifest, 'id', (100, 100))


class MockSAM3Tests(unittest.TestCase):
    def test_real_decode_interface_is_full_twenty_class_no_preview(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is tested in the target moduseg_sam3 CPU environment')
        from types import SimpleNamespace
        calls = dict(encoder=0, decoder=[])

        class Processor:
            def set_image(self, image):
                calls['encoder'] += 1
                self.assertion = image.size
                return {'frozen': True}

        class Proposer:
            device = torch.device('cpu')
            processor = Processor()

            @staticmethod
            def _buckets(indices):
                return [indices[i:i+4] for i in range(0, len(indices), 4)]

            @staticmethod
            def _ground(state, indices, expand):
                calls['decoder'].extend(indices)
                assert state['frozen'] and expand
                # Normal presence .1 makes a high object probability fail the .15 rule.
                presence = torch.ones((4, 1)) * .9
                presence[0] = .1
                logits = torch.full((4, 2, 1), 5.)
                masks = torch.full((4, 2, 8, 8), -8.)
                masks[:, :, 1:7, 1:7] = 8.
                return dict(presence_logit_dec=torch.logit(presence), pred_logits=logits, pred_masks=masks)

        arrays, report = r.decode_image(Proposer(), np.zeros((40, 40, 3), np.uint8), 'id', 'R1')
        self.assertEqual(calls['encoder'], 1)
        self.assertEqual(calls['decoder'], list(range(20)))
        self.assertEqual(report['decoder_bucket_calls'], 5)
        self.assertEqual(report['preview_calls'], 0)
        self.assertFalse(report['force_presence_one'])
        self.assertEqual(report['raw_threshold_candidates'], 30)
        self.assertTrue(set(arrays['source_classes']).issubset(set(range(2, 21))))


if __name__ == '__main__':
    unittest.main()
