import torch
from torch.nn import functional as F


def single_query(att, query, key, value, mask):
    if query.shape[-2] != 1:
        raise ValueError("Expected one query")
    h, d = att.heads, att.dim // att.heads
    q = att.query(query).unflatten(-1, (h, d))
    k = key.unflatten(-1, (h, d))
    v = value.unflatten(-1, (h, d))
    safe, present = mask
    valid = safe.reshape(*query.shape[:-2], key.shape[-2], 1)
    scores = (q.float() * k.float()).sum(-1) * (d ** -0.5)
    weights = scores.masked_fill(~valid, -torch.inf).softmax(-2)
    weights = F.dropout(weights, att.dropout, att.training)
    msg = (weights.unsqueeze(-1) * v.float()).sum(-3).to(q.dtype).reshape_as(query)
    return att.output(msg) * present.reshape(*query.shape[:-2], 1, 1)


def one_card(att, query, key, value, mask):
    if key.shape[-2] != 1:
        raise ValueError("Expected exactly one region card")
    h, d = att.heads, att.dim // att.heads
    q = att.query(query).unflatten(-1, (h, d))
    k = key.unflatten(-1, (h, d))
    v = value.unflatten(-1, (h, d))
    # Preserve zero (rather than absent) gradients for query/key projections.
    weights = 1 + (q[..., 0].float() + k[..., 0].float()) * 0
    weights = F.dropout(weights, att.dropout, att.training)
    msg = (weights.unsqueeze(-1) * v.float()).to(q.dtype).reshape_as(query)
    return att.output(msg) * mask[1].reshape(*query.shape[:-2], 1, 1)


def pool_forward(att, query, source, valid):
    from src.attention_helpers import prepare_mask
    k, v = att.project_kv(source)
    return single_query(att, query, k, v, prepare_mask(valid))
