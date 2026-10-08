"""Experimental feature-only SAM3 resolution adaptation; no checkpoint edits."""
import torch
from types import MethodType
import torch.nn.functional as F


def chunked_query_attention(q, k, v, chunk_size):
    """Inference SDPA: partition query rows, retain all keys for each softmax."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError('Query chunking is an inference-only implementation')
    output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    for start in range(0, q.shape[-2], chunk_size):
        end = min(start + chunk_size, q.shape[-2])
        output[..., start:end, :] = F.scaled_dot_product_attention(q[..., start:end, :], k, v)
    return output


def _chunked_forward(self, x):
    if self.cls_token or self.use_rel_pos or x.ndim not in (3, 4):
        raise ValueError('Unsupported SAM3 attention layout for query chunking')
    batch = x.shape[0]
    length = x.shape[1] * x.shape[2] if x.ndim == 4 else x.shape[1]
    qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = self._apply_rope(q, k)
    result = chunked_query_attention(q, k, v, self._query_chunk_size)
    result = result.transpose(1, 2).reshape(*x.shape[:-1], -1)
    return self.proj(result)


def enable_query_chunking(trunk, chunk_size):
    if chunk_size < 0:
        raise ValueError('Query chunk size must be nonnegative')
    if chunk_size == 0:
        return
    targets = [b.attn for b in trunk.blocks if b.window_size == 0]
    for attn in targets:
        if (getattr(attn.attn_type, 'name', attn.attn_type) != 'Vanilla' or attn.use_rel_pos or attn.cls_token
                or not callable(getattr(attn, '_apply_rope', None))):
            raise ValueError('Unsupported installed SAM3 global attention; cannot enable query chunking')
    for attn in targets:
        attn._query_chunk_size = chunk_size
        attn.forward = MethodType(_chunked_forward, attn)


def adapt_feature_resolution(trunk, resolution):
    # Preserve the native model and its historical signatures exactly.
    if resolution == 1008:
        return None
    if resolution < 336 or resolution % 336:
        raise ValueError('Experimental SAM3 resolution must be a multiple of 336 (e.g. 1344, 2016)')
    projection = trunk.patch_embed.proj
    if tuple(projection.kernel_size) != (14, 14) or tuple(projection.stride) != (14, 14):
        raise ValueError('Unsupported SAM3 patch embedding; expected stride/kernel 14')
    if getattr(trunk, 'retain_cls_token', False) or not getattr(trunk, 'tile_abs_pos', False):
        raise ValueError('Unsupported SAM3 class-token/absolute-position configuration')
    global_attention = []
    for block in trunk.blocks:
        attn = block.attn
        if block.window_size:
            if resolution // 14 % block.window_size:
                raise ValueError('Requested grid does not tile local attention windows')
            continue
        if (not attn.use_rope or attn.use_rel_pos or attn.use_ve_rope or attn.rope_tiled
                or not attn.rope_interp or not callable(getattr(attn, '_setup_rope_freqs', None))):
            raise ValueError('Unsupported global attention implementation for resolution adaptation')
        if tuple(attn.input_size) != (72, 72):
            raise ValueError('Expected unmodified native 72x72 global attention grid')
        global_attention.append(attn)
    if not global_attention:
        raise ValueError('No SAM3 global attention blocks found')
    for attn in global_attention:
        device = attn.freqs_cis.device
        attn.input_size = (resolution // 14, resolution // 14)
        # Keep rope_pt_size/theta and learned weights unchanged. Regenerate the
        # official interpolated frequencies for the new global grid only.
        attn._setup_rope_freqs()
        for name in ('freqs_cis', 'freqs_cis_real', 'freqs_cis_imag'):
            value = getattr(attn, name, None)
            if isinstance(value, torch.Tensor):
                setattr(attn, name, value.to(device=device))
        if attn.freqs_cis.shape[0] != (resolution // 14) ** 2:
            raise RuntimeError('Regenerated RoPE grid has an unexpected size')
    return dict(version=1, mode='feature_only_global_rope_interp', native_resolution=1008,
                resolution=resolution, global_blocks=len(global_attention))
