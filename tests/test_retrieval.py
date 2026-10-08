import unittest

import numpy as np

from src.retrieval import build_retrieval_context, classify_queries


class RetrievalTests(unittest.TestCase):
    def test_balanced_scoring_reduces_gallery_count_bias(self):
        foreground = np.array([[1.0, 0.0]], dtype=np.float32)
        background = np.repeat(
            np.array([[0.9, np.sqrt(1.0 - 0.9**2)]], dtype=np.float32),
            20,
            axis=0,
        )
        gallery = np.vstack([foreground, background])
        labels = ["dog"] + ["background"] * len(background)
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        context = build_retrieval_context(gallery, labels)

        legacy = classify_queries(
            query, gallery, labels, None, context, topk=5, mode="legacy"
        )
        balanced = classify_queries(
            query,
            gallery,
            labels,
            None,
            context,
            topk=5,
            mode="balanced",
            class_topm=3,
            search_multiplier=2,
            centroid_weight=0.4,
        )

        self.assertEqual(legacy[0].label, "background")
        self.assertEqual(balanced[0].label, "dog")
        self.assertGreater(balanced[0].margin, 0.0)

    def test_background_variants_are_one_class(self):
        gallery = np.eye(3, dtype=np.float32)
        labels = ["background_0", "background_1", "cat"]
        context = build_retrieval_context(gallery, labels)

        self.assertEqual(context.labels, ("background", "cat"))
        self.assertEqual(context.counts.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
