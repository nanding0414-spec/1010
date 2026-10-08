import unittest
from types import SimpleNamespace
import torch
from src.sam3_resolution import adapt_feature_resolution


class Attention(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2,2))
        self.input_size = size
        self.rope_pt_size = (24,24)
        self.use_rope = self.rope_interp = True
        self.use_rel_pos = self.use_ve_rope = self.rope_tiled = False
        self.calls = 0
        self._setup_rope_freqs()

    def _setup_rope_freqs(self):
        self.calls += 1
        self.register_buffer('freqs_cis', torch.ones(self.input_size[0]*self.input_size[1],2,dtype=torch.complex64))
        self.register_buffer('freqs_cis_real', self.freqs_cis.real)
        self.register_buffer('freqs_cis_imag', self.freqs_cis.imag)


def trunk():
    return SimpleNamespace(patch_embed=SimpleNamespace(proj=torch.nn.Conv2d(3,2,14,stride=14)),
        retain_cls_token=False, tile_abs_pos=True, blocks=[SimpleNamespace(window_size=24,attn=Attention((24,24))),
                                                        SimpleNamespace(window_size=0,attn=Attention((72,72)))])


class ResolutionTests(unittest.TestCase):
    def test_native_noop_and_highres_preserves_weights_and_windows(self):
        model=trunk(); local, glob=[b.attn for b in model.blocks]
        original=glob.weight.detach().clone()
        self.assertIsNone(adapt_feature_resolution(model,1008))
        self.assertEqual(glob.calls,1)
        signature=adapt_feature_resolution(model,2016)
        self.assertEqual(glob.input_size,(144,144))
        self.assertEqual(glob.freqs_cis.shape[0],20736)
        self.assertEqual(glob.freqs_cis_real.shape,glob.freqs_cis.shape)
        self.assertEqual(glob.rope_pt_size,(24,24))
        self.assertEqual(local.calls,1)
        self.assertTrue(torch.equal(original,glob.weight))
        self.assertEqual(signature['global_blocks'],1)

    def test_unsupported_architecture_fails_before_mutation(self):
        model=trunk(); model.blocks[-1].attn.use_rel_pos=True
        with self.assertRaises(ValueError):
            adapt_feature_resolution(model,2016)
        self.assertEqual(model.blocks[-1].attn.input_size,(72,72))
        with self.assertRaises(ValueError):
            adapt_feature_resolution(trunk(),2000)


if __name__ == '__main__':
    unittest.main()
