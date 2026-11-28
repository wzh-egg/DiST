"""Direct global card attention with soft learned relative-position bias.

    same v1 summaries -> all-card attention -> same grid write -> same station update

No hard distance cutoff, top-k, extra global token, dual-state fusion, new
calendar input, or future input. Four rounds remain independently parameterized.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.layers import HierarchicalInteraction
from src.attention_helpers import prepare_mask, attend_prepared, project_gather


class RelativePositionBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        # A deliberately weak initial preference: at 1000 km the logit penalty
        # is about -0.24, not an exclusion. Each head/round can adapt its strength.
        self.distance_raw = nn.Parameter(torch.full((heads,), math.log(math.expm1(.1))))
        # No head-wise constant output bias: a constant across all source keys
        # cancels in softmax and would only accumulate numerical noise in Adam.
        self.net = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads, bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, relative):
        return (self.net(relative) - relative[..., 2, None]*F.softplus(self.distance_raw)).permute(2, 0, 1)


def card_attention(update, cards, valid, bias, literal=False):
    """Simultaneous full self-read; distinct original Q/KV LayerNorms retained.

    Literal route independently materializes dot products/softmax, used to
    check outputs and all gradients against SDPA with additive float bias.
    Empty key sets have a dummy key and a zeroed complete message.
    """
    attention = update.attention
    b, count, dim = cards.shape
    heads, width = attention.heads, dim//attention.heads
    q = attention.query(update.q_norm(cards)).reshape(b, count, heads, width).transpose(1, 2)
    source = update.kv_norm(cards)
    k, v = attention.project_kv(source)
    k = k.reshape(b, count, heads, width).transpose(1, 2)
    v = v.reshape(b, count, heads, width).transpose(1, 2)
    safe, available = prepare_mask(valid)
    mask = bias[None].to(q.dtype).expand(b, -1, -1, -1).masked_fill(~safe[:, None, None, :], -torch.inf)
    dropout = attention.dropout if attention.training else 0.
    if literal:
        # Reference emulates SDPA's higher-precision intermediates under AMP,
        # instead of rounding logits and the softmax to fp16 at each operator.
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
            weights = ((q.to(dtype) @ k.to(dtype).transpose(-2, -1))/math.sqrt(width) + mask.to(dtype)).softmax(-1)
            result = (F.dropout(weights, dropout, training=attention.training) @ v.to(dtype)).to(q.dtype)
    else:
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
    message = attention.output(result.transpose(1, 2).flatten(-2))
    message = message * available[:, None, None].to(message.dtype)
    result = update.ffn(cards + update.drop(message))
    return result * valid[..., None].to(result.dtype)


class DirectCardRound(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Keep every non-bridge original Parameter under its original name.
        # regional_read is now the one all-card update. global_read and its
        # learned seeds are genuinely absent, not computed and discarded.
        for name, module in original.named_children():
            if name != 'global_read':
                self.add_module(name, module)
        self.spatial_bias = RelativePositionBias(original.regional_read.attention.heads)

    def step(self, station, grid, grid_valid, geometry, masks, kv_mode,
             details=False, literal=False, position_bias=True):
        regional, regional_valid = self.regional(station, grid, grid_valid, geometry)
        flat = regional.flatten(1, 2)
        bias = self.spatial_bias(geometry.card_relative_features)
        if not position_bias:
            bias = torch.zeros_like(bias)
        updated = card_attention(self.regional_read, flat, regional_valid.flatten(1, 2), bias, literal)
        regional = updated.reshape_as(regional)
        if literal:
            keys, values = self.grid_read.attention.project_kv(self.grid_read.kv_norm(regional))
            keys, values = keys[:, geometry.block_region], values[:, geometry.block_region]
            message = self.grid_read.attention.attend(self.grid_read.q_norm(grid), keys, values,
                                                     regional_valid[:, geometry.block_region])
        else:
            keys, values = project_gather(self.grid_read.attention, self.grid_read.kv_norm(regional),
                                         geometry.block_region, False, kv_mode)
            message = attend_prepared(self.grid_read.attention, self.grid_read.q_norm(grid), keys, values,
                                      prepare_mask(regional_valid[:, geometry.block_region]))
        new_grid = self.grid_read.ffn(grid + self.grid_read.drop(message))
        new_grid = new_grid * grid_valid[..., None].to(new_grid.dtype)
        s = self.station_norm(station)
        if literal:
            keys, values = self.station_attention.project_kv(s)
            keys, values = keys[:, geometry.station_neighbors], values[:, geometry.station_neighbors]
            smask = geometry.station_neighbor_valid[None].expand(len(s), -1, -1)
            station_message = self.station_attention.attend(s[:, :, None], keys, values, smask).squeeze(-2)
            keys, values = self.grid_attention.project_kv(self.grid_norm(new_grid))
            keys = keys[:, geometry.candidate_to_unique].flatten(2, 3)
            values = values[:, geometry.candidate_to_unique].flatten(2, 3)
            gmask = (grid_valid[:, geometry.candidate_to_unique] & geometry.candidate_valid[None, :, :, None]).flatten(2, 3)
            grid_message = self.grid_attention.attend(s[:, :, None], keys, values, gmask).squeeze(-2)
        else:
            keys, values = project_gather(self.station_attention, s, geometry.station_neighbors, False, kv_mode)
            station_message = attend_prepared(self.station_attention, s[:, :, None], keys, values, masks[0]).squeeze(-2)
            keys, values = project_gather(self.grid_attention, self.grid_norm(new_grid), geometry.candidate_to_unique, True, kv_mode)
            grid_message = attend_prepared(self.grid_attention, s[:, :, None], keys, values, masks[1]).squeeze(-2)
        new_station = self.station_ffn(station + self.station_drop(self.station_fuse(torch.cat((station_message, grid_message), -1))))
        detail = dict(station=new_station, grid=new_grid, regional=regional,
                      regional_valid=regional_valid, position_bias=bias) if details else None
        return new_station, new_grid, detail


class CardAttentionInteraction(HierarchicalInteraction):
    def __init__(self, geometry, kv_mode='packed', position_bias=True, **kwargs):
        if 'card_relative_features' not in geometry or 'card_coordinates_lon_lat' not in geometry:
            raise ValueError('Frozen value-free card coordinates are required.')
        if kv_mode not in ('packed', 'separate'):
            raise ValueError('Invalid K/V mode.')
        # Construct the WHOLE original v1 first. This preserves all retained
        # initial weights, including the prediction head, under the same seed.
        super().__init__(geometry, **kwargs)
        self.interactions = nn.ModuleList(DirectCardRound(block) for block in self.interactions)
        self.kv_mode, self.position_bias = kv_mode, position_bias
        expected = len(geometry['region_station_ids'])*(kwargs.get('subgroups', 8)+kwargs.get('station_tokens', 2))
        if geometry['card_relative_features'].shape != (expected, expected, 3):
            raise ValueError('Card position order/shape mismatch.')

    def _forward(self, station_input, patches, block_valid, return_details, literal):
        if station_input.shape[1:] != (2, self.nodes, 12):
            raise ValueError('Station input must be (B,2,N,12).')
        if patches.shape != (len(station_input), self.blocks, 12, 2, self.patch_size, self.patch_size):
            raise ValueError('Grid input/order/shape differs from frozen geometry.')
        if block_valid.shape != patches.shape[:2] or block_valid.dtype != torch.bool:
            raise ValueError('Invalid block mask.')
        station = self.station_encoder(station_input)
        grid, valid = self.grid_encoder(patches, block_valid)
        geometry, b = self.geometry, len(station_input)
        smask = geometry.station_neighbor_valid[None].expand(b, -1, -1)
        gmask = (valid[:, geometry.candidate_to_unique] & geometry.candidate_valid[None, :, :, None]).flatten(2, 3)
        masks = prepare_mask(smask), prepare_mask(gmask)
        trace = []
        for block in self.interactions:
            station, grid, detail = block.step(station, grid, valid, geometry, masks, self.kv_mode,
                                               return_details, literal, self.position_bias)
            if return_details:
                trace.append(detail)
        prediction = self.prediction_head(station).transpose(1, 2).contiguous()
        if return_details:
            return prediction, dict(rounds=trace, station=station, grid=grid, grid_valid=valid)
        return prediction

    def forward(self, station_input, patches, block_valid, return_details=False):
        return self._forward(station_input, patches, block_valid, return_details, False)
