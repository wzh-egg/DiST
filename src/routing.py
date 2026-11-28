"""Mix distributions, not logits. Both arms have exactly identical initial values."""
import json
import math
from pathlib import Path
import torch
from torch import nn


def masked_probability(logits, eligible):
    mask = eligible[:, None]
    present = mask.any(-1, keepdim=True)
    safe = torch.where(present, logits.masked_fill(~mask, -torch.inf), 0.)
    return safe.softmax(-1) * mask


class RoutingScore(nn.Module):
    # Preserve the established module import/name; the formula is a probability mixture.
    def __init__(self, heads=4, arm=None, initial_fraction=None):
        super().__init__()
        job = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'air.json').read_text())
        if arm is None:
            arm = job['arm']
        if initial_fraction is None:
            initial_fraction = job.get('content_fraction_initial', .1)
        if not 0 < initial_fraction < 1:
            raise ValueError('Initial content fraction must be in (0,1)')
        if arm not in ('fixed', 'learned'):
            raise ValueError(arm)
        self.arm = arm
        self.geo_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads, bias=False))
        nn.init.zeros_(self.geo_mlp[-1].weight)
        initial = torch.full((heads,), math.log(initial_fraction / (1 - initial_fraction)))
        if arm == 'learned':
            self.mix_logit = nn.Parameter(initial)
        else:
            self.register_buffer('mix_logit', initial)

    def content_fraction(self):
        return self.mix_logit.sigmoid()

    def forward(self, q, k, relative, eligible):
        content = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        geo = self.geo_mlp(relative.float()).permute(2, 0, 1)[None].expand_as(content)
        pc = masked_probability(content, eligible)
        pg = masked_probability(geo, eligible)
        lam = self.content_fraction()[None, :, None, None]
        probability = (1 - lam) * pg + lam * pc
        return probability, (content, geo, pc, pg)
