"""DiST for hourly station-level AQI forecasting."""
import json
from pathlib import Path

import torch
from torch import nn

from src.attention_helpers import prepare_mask
from src.dual_path import DualPathModel
from src.region_exchange import RegionExchangeRound


class DiST(DualPathModel):
    """Four rounds of local and global interaction with horizon-specific fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions = nn.ModuleList(
            RegionExchangeRound(block, index)
            for index, block in enumerate(self.interactions)
        )
        config = json.loads((Path(__file__).parent / "configs" / "air.json").read_text())
        initial = torch.tensor(config["horizon_alpha_initial"], dtype=torch.float32)
        if initial.shape != (12,) or not ((initial > 0) & (initial < 1)).all():
            raise ValueError("Expected 12 fusion weights between zero and one")
        self.horizon_logits = nn.Parameter(torch.logit(initial))

    def gates(self, local, global_state):
        return self.horizon_logits.sigmoid()[None, None, :, None].expand(
            *local.shape[:2], 12, self.hidden
        )

    def forward(self, x, patches, block_valid, return_details=False):
        s0 = self.station_encoder(x)
        g0, valid = self.grid_encoder(patches, block_valid)
        geo = self.geometry
        masks = (prepare_mask(geo.station_neighbor_valid[None].expand(len(s0), -1, -1)), prepare_mask((valid[:, geo.candidate_to_unique] & geo.candidate_valid[None, :, :, None]).flatten(2, 3)))
        paths = []
        traces = []
        for external in (False, True):
            station, grid = (s0, g0)
            trace = []
            for block in self.interactions:
                station, grid, keep = block.step(station, grid, valid, geo, masks, self.kv_mode, external, return_details)
                if return_details:
                    trace.append(keep)
            paths.append(station)
            traces.append(trace)
        local, global_state = paths
        gates = self.gates(local, global_state)
        mixed = (1 - gates) * local.unsqueeze(-2) + gates * global_state.unsqueeze(-2)
        hidden = self.prediction_head[1](self.prediction_head[0](mixed))
        head = self.prediction_head[2]
        y = ((hidden * head.weight).sum(-1) + head.bias).transpose(1, 2).contiguous()
        return (y, dict(gates=gates, local=local, global_state=global_state, routes=traces)) if return_details else y
