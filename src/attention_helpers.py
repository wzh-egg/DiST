"""Exact attention execution helpers: packed or separate projected K/V."""
import torch
from torch.nn import functional as F

def prepare_mask(valid):
    flat = valid.reshape(-1, valid.shape[-1])
    available = flat.any(-1)
    first = torch.arange(flat.shape[-1], device=flat.device) == 0
    safe = flat | ((~available)[..., None] & first)
    return safe, available


def attend_prepared(attention, query, key, value, mask):
    """Original SDPA, with the identical candidate mask prepared once per forward."""
    safe, available = mask
    shape = query.shape
    h, d = attention.heads, attention.dim // attention.heads
    q = attention.query(query).reshape(-1, shape[-2], h, d).transpose(1, 2)
    k = key.reshape(-1, key.shape[-2], h, d).transpose(1, 2)
    v = value.reshape(-1, value.shape[-2], h, d).transpose(1, 2)
    pieces = []
    for start in range(0, q.shape[0], attention.batch_chunk):
        end = start + attention.batch_chunk
        pieces.append(F.scaled_dot_product_attention(
            q[start:end], k[start:end], v[start:end],
            # prepare_mask makes the sole key safe even for empty rows;
            # available below still zeros empty rows after the output projection.
            # Avoid a redundant singleton bias with zero strides in CUDA backward.
            attn_mask=None if safe.shape[-1] == 1 else safe[start:end, None, None, :],
            dropout_p=attention.dropout if attention.training else 0.0))
    result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, 0)
    result = attention.output(result.transpose(1, 2).flatten(-2))
    return (result * available[..., None, None].to(result.dtype)).reshape(shape)


def project_gather(attention, source, ids, flatten_history, mode):
    if mode == 'packed':
        # One GEMM and one address gather instead of separate K and V operations.
        # Cat is differentiable; key/value remain separate original Parameters.
        weights = torch.cat((attention.key.weight, attention.value.weight), 0)
        packed = F.linear(source, weights)[:, ids]
        if flatten_history:
            packed = packed.flatten(2, 3)
        return packed.chunk(2, -1)
    keys, values = attention.project_kv(source)
    keys, values = keys[:, ids], values[:, ids]
    if flatten_history:
        keys, values = keys.flatten(2, 3), values.flatten(2, 3)
    return keys, values


