import unittest

import numpy as np

try:
    from src.mask_proposer_sam3 import _nms_by_masks
    from src.mask_proposer_sam3_query import parse_query_prompts
except Exception:  # Lightweight local runtime may not have torch/SAM3 installed.
    _nms_by_masks = None
    parse_query_prompts = None


@unittest.skipUnless(parse_query_prompts is not None, "requires torch and SAM3")
class MaskProposerTests(unittest.TestCase):
    def test_query_prompt_parser_deduplicates_and_strips(self):
        self.assertEqual(
            parse_query_prompts(" object | thing | object "),
            ["object", "thing"],
        )

    def test_mask_nms_keeps_distinct_masks_with_same_bbox(self):
        diagonal = np.eye(5, dtype=bool)
        anti_diagonal = np.fliplr(diagonal)
        keep = _nms_by_masks([diagonal, anti_diagonal], [0.9, 0.8], 0.5)
        self.assertEqual(keep, [0, 1])

    def test_mask_nms_suppresses_duplicate(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        keep = _nms_by_masks([mask, mask.copy()], [0.8, 0.9], 0.7)
        self.assertEqual(keep, [1])


if __name__ == "__main__":
    unittest.main()
