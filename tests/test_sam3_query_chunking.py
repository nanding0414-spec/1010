from types import SimpleNamespace
from enum import Enum
import unittest
import torch
import torch.nn.functional as F
from src.sam3_resolution import chunked_query_attention, enable_query_chunking


class AttentionType(Enum):
    Vanilla = 0


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_type = 'Vanilla'
        self.cls_token = self.use_rel_pos = False
        self.num_heads = 2
        self.qkv = torch.nn.Linear(8,24)
        self.proj = torch.nn.Linear(8,8)

    def _apply_rope(self,q,k):
        # Nonidentity transform verifies it is still called before chunking.
        return q.flip(-1), k.flip(-1)

    def forward(self,x):
        q,k,v = self.qkv(x).reshape(x.shape[0],-1,3,2,4).permute(2,0,3,1,4).unbind(0)
        q,k = self._apply_rope(q,k)
        y=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(*x.shape[:-1],8)
        return self.proj(y)


class QueryChunkTests(unittest.TestCase):
    def test_sdpa_equivalence_with_tail_and_all_keys(self):
        torch.manual_seed(1)
        q,k,v=[torch.randn(2,3,19,8) for _ in range(3)]
        expected=F.scaled_dot_product_attention(q,k,v)
        for size in (1,7,19,32):
            torch.testing.assert_close(chunked_query_attention(q,k,v,size),expected,atol=1e-6,rtol=1e-5)

    def test_global_forward_equivalence_and_local_unchanged(self):
        torch.manual_seed(4)
        global_attn,local_attn=Attention().eval(),Attention().eval()
        trunk=SimpleNamespace(blocks=[SimpleNamespace(window_size=0,attn=global_attn),
                                      SimpleNamespace(window_size=24,attn=local_attn)])
        with torch.inference_mode():
            inputs=[torch.randn(1,3,5,8),torch.randn(2,17,8)]
            expected=[global_attn(x) for x in inputs]
            enable_query_chunking(trunk,4)
            for x,y in zip(inputs,expected):
                torch.testing.assert_close(global_attn(x),y,atol=1e-6,rtol=1e-5)
        self.assertNotIn('forward',local_attn.__dict__)

    def test_reject_training_and_bad_chunk(self):
        q=torch.randn(1,2,3,4,requires_grad=True)
        with self.assertRaises(RuntimeError):
            chunked_query_attention(q,q,q,2)
        with self.assertRaises(ValueError):
            chunked_query_attention(q,q,q,0)


if __name__ == '__main__':
    unittest.main()
