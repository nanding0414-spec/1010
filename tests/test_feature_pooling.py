import unittest

import numpy as np

try:
    import torch
    from src.feature_extractor import BaseFeatureExtractor
except Exception:  # Local lightweight test runtime may not have torch installed.
    torch = None
    BaseFeatureExtractor = None


if BaseFeatureExtractor is not None:
    class DummySpatialExtractor(BaseFeatureExtractor):
        def __init__(self):
            super().__init__(
                device=torch.device("cpu"),
                patch_size=(2, 2),
                embed_dim=4,
                model_prefix="dummy",
                norm_mean=(0.0, 0.0, 0.0),
                norm_std=(1.0, 1.0, 1.0),
            )

        def _forward_tokens(self, inp):
            tokens = torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            )
            return tokens, (2, 2)


@unittest.skipUnless(torch is not None, "requires torch")
class FeaturePoolingTests(unittest.TestCase):
    def test_single_and_batch_pooling_match(self):
        extractor = DummySpatialExtractor()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        single = extractor.extract_features(image, mask).numpy()
        batch = extractor.extract_features_batch(image, [mask])[0]
        np.testing.assert_allclose(single, batch, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(single, [1.0, 0.0, 0.0, 0.0], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
