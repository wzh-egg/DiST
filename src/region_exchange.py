"""RIGU with masked probability mixture for joint selection and reading."""
import torch
from torch import nn
from torch.nn import functional as F
from src.dual_path import CellRound
from src.routing import RoutingScore


class RegionExchangeRound(CellRound):
    def __init__(self, old, index):
        nn.Module.__init__(self)
        for name, module in old.named_children():
            if name != 'spatial_bias':
                self.add_module(name, module)
        # Preserve every retained original backbone tensor and its RNG stream.
        with torch.random.fork_rng(devices=[]):
            state = torch.Generator(device='cpu').manual_seed(420100 + index).get_state()
            torch.set_rng_state(state)
            self.scorer = RoutingScore(self.regional_read.attention.heads)

    def exchange(self, cards, valid, geo, external):
        update = self.regional_read
        a = update.attention
        def heads(x):
            return x.unflatten(-1, (a.heads, -1)).transpose(1, 2)
        # Keep the regional projections/scoring/aggregation in FP32 also on AIR.
        with torch.autocast(device_type=cards.device.type, enabled=False):
            x = cards.float()
            q = heads(a.query(update.q_norm(x)))
            k, v = a.project_kv(update.kv_norm(x))
            k, v = heads(k), heads(v)
            eligible = valid[:, None, :].expand(-1, cards.shape[1], -1)
            if external:
                eye = torch.eye(cards.shape[1], device=cards.device, dtype=torch.bool)
                eligible = eligible & ~eye[None]
            else:
                eligible = eligible & geo.cell_adjacent[None]
            probabilities, _ = self.scorer(q, k, geo.cell_relative, eligible)
            if external:
                ranking = probabilities.detach().mean(1).masked_fill(~eligible, -torch.inf)
                ids = ranking.topk(min(50, cards.shape[1] - 1), dim=-1).indices
                keep = eligible & torch.zeros_like(eligible).scatter_(-1, ids, True)
            else:
                keep = eligible
            available = keep.any(-1)
            selected = probabilities * keep[:, None]
            # Restrict the mixed probability distribution; do NOT softmax probabilities.
            weights = selected / selected.sum(-1, keepdim=True).clamp_min(1e-30)
            applied = F.dropout(weights, a.dropout, training=a.training)
            combined = applied @ v
            message = a.output(combined.transpose(1, 2).reshape_as(x))
            message = message * available[..., None]
            result = update.ffn(x + update.drop(message)) * valid[..., None]
        return result, keep
