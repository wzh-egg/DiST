"""Four-round station/grid interaction through regional and fixed global tokens.

    region summaries -> global bridge -> region summaries -> grid -> stations

Station updates use old neighboring station states and newly updated grid
states. There is no region-to-station write. Grid stage order is maintained.
No node ID, geographic embedding, calendar embedding or time-gap bias is used.
The existing numeric hour/24 input channel is retained.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.temporal import causal_conv3

SDPA_BATCH_CHUNK = 8192


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.net(self.norm(x)))


class Attention(nn.Module):
    """Projected K/V may be shared before station/block-specific gathering."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError('Hidden size must be divisible by heads.')
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.batch_chunk = SDPA_BATCH_CHUNK
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def project_kv(self, source):
        return self.key(source), self.value(source)

    def attend(self, query, key, value, valid):
        h, d = self.heads, self.dim // self.heads
        shape = query.shape
        # Flatten batch/station axes so CUDA SDPA receives its 4D layout;
        # keeping a 5D tensor here can force a much larger math fallback.
        q = self.query(query).reshape(-1, shape[-2], h, d).transpose(1, 2)
        k = key.reshape(-1, key.shape[-2], h, d).transpose(1, 2)
        v = value.reshape(-1, value.shape[-2], h, d).transpose(1, 2)
        valid = valid.reshape(-1, valid.shape[-1])
        available = valid.any(-1)
        # Supply a safe dummy key for empty sets, then zero the complete message.
        first = torch.arange(valid.shape[-1], device=valid.device) == 0
        safe = valid | ((~available)[..., None] & first)
        # torch2.5 CUDA efficient/flash attention can map batch directly to a
        # CUDA grid axis limited to65535. B32*U2568=82176 exceeds that limit.
        # Chunk ONLY independent attention batches. Every query still sees its
        # complete key/value set; never split the history/key softmax axis.
        pieces = []
        for start in range(0, q.shape[0], self.batch_chunk):
            stop = start + self.batch_chunk
            pieces.append(F.scaled_dot_product_attention(
                q[start:stop], k[start:stop], v[start:stop],
                attn_mask=None if safe.shape[-1] == 1 else safe[start:stop, None, None, :],
                dropout_p=self.dropout if self.training else 0.0))
        result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        result = result.transpose(1, 2).flatten(-2)
        result = self.output(result)
        result = result * available[..., None, None].to(result.dtype)
        return result.reshape(shape)

    def forward(self, query, source, valid):
        return self.attend(query, *self.project_kv(source), valid)


class CrossUpdate(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.q_norm, self.kv_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        self.drop, self.ffn = nn.Dropout(dropout), FeedForward(dim, dropout)

    def forward(self, query, source, valid, query_valid=None):
        x = query + self.drop(self.attention(self.q_norm(query), self.kv_norm(source), valid))
        x = self.ffn(x)
        if query_valid is not None:
            x = x * query_valid[..., None].to(x.dtype)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, dim, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, dilation=dilation)
        self.dilation, self.drop = dilation, nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(F.relu(causal_conv3(x, self.conv)))


class StationEncoder(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.project = nn.Conv1d(2, dim, 1)
        self.temporal = nn.Sequential(*(TemporalBlock(dim, d, dropout) for d in (1, 2, 4)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, _, n, t = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(b*n, 2, t)
        return self.norm(self.temporal(self.project(sequence))[:, :, -1]).reshape(b, n, -1)


class GridEncoder(nn.Module):
    """CNN per frame, ordered 3-frame projection, GRU over four history stages."""
    def __init__(self, dim, patch_size=5, chunk_blocks=8192):
        super().__init__()
        self.dim, self.patch_size, self.chunk_blocks = dim, patch_size, chunk_blocks
        self.cnn = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.frame = nn.Sequential(nn.Flatten(), nn.Linear(32*patch_size**2, dim), nn.ReLU())
        self.segment = nn.Sequential(nn.Linear(3*dim, dim), nn.GELU())
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, patches, block_valid):
        b, u, t, c, p, _ = patches.shape
        sequence = patches.reshape(b*u, t, c, p, p)
        encoded, masks = [], []
        for start in range(0, b*u, self.chunk_blocks):
            part = sequence[start:start+self.chunk_blocks]
            valid_pixels = part[:, :, 1] > 0.5
            aq = torch.where(valid_pixels, part[:, :, 0], 0.0)
            frames = torch.stack((aq, valid_pixels.to(part.dtype)), 2).flatten(0, 1)
            valid_frames = valid_pixels.any(-1).any(-1)
            features = self.frame(self.cnn(frames)).reshape(len(part), 12, self.dim)
            features = features * valid_frames[..., None].to(features.dtype)
            segment_valid = valid_frames.reshape(len(part), 4, 3).any(-1)
            features = self.segment(features.reshape(len(part), 4, 3*self.dim))
            features = features * segment_valid[..., None].to(features.dtype)
            states, _ = self.gru(features)
            # GRU states contain prefixes, so a missing later segment can still
            # carry preceding observations. No-observation prefixes stay masked.
            prefix_valid = segment_valid.long().cumsum(-1) > 0
            encoded.append(states * prefix_valid[..., None].to(states.dtype))
            masks.append(prefix_valid)
        valid = torch.cat(masks).reshape(b, u, 4) & block_valid[..., None]
        states = torch.cat(encoded).reshape(b, u, 4, self.dim)
        return states * valid[..., None].to(states.dtype), valid


class RegionalSummaries(nn.Module):
    def __init__(self, dim, heads, dropout, subgroups, station_tokens):
        super().__init__()
        self.subgroups, self.station_tokens = subgroups, station_tokens
        self.spatial_score = nn.Sequential(nn.Linear(4*dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.history_card = nn.Sequential(nn.Linear(7*dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.station_seeds = nn.Parameter(torch.randn(station_tokens, dim) / math.sqrt(dim))
        self.station_norm, self.seed_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.station_pool = Attention(dim, heads, dropout)

    def forward(self, station, grid, grid_valid, geometry):
        b, _, _, d = grid.shape
        ids, padding = geometry.group_ids, geometry.group_valid
        grouped = grid[:, ids]                       # B,(G*A),C,4,D
        valid = grid_valid[:, ids] & padding[None, :, :, None]
        score = self.spatial_score(grouped.flatten(-2)).squeeze(-1).float()
        block_valid = valid.any(-1)
        safe_score = score.masked_fill(~block_valid, -1e4)
        alpha = safe_score.softmax(-1) * block_valid.to(score.dtype)
        # Common spatial scores for all stages; renormalize only to exclude
        # unavailable history prefixes, never treat padding as observations.
        weights = alpha[..., None] * valid.to(alpha.dtype)
        weights = weights / weights.sum(-2, keepdim=True).clamp_min(1e-12)
        history = (weights[..., None].to(grouped.dtype) * grouped).sum(-3)
        delta = history[:, :, 1:] - history[:, :, :-1]
        card = self.history_card(torch.cat((history.flatten(-2), delta.flatten(-2)), -1))
        card_valid = valid.any(-1).any(-1)
        g = geometry.region_station_ids.shape[0]
        card = card.reshape(b, g, self.subgroups, d)
        card_valid = card_valid.reshape(b, g, self.subgroups)
        card = card * card_valid[..., None].to(card.dtype)

        sids = geometry.region_station_ids
        smask = geometry.region_station_valid[None].expand(b, -1, -1)
        source = self.station_norm(station)[:, sids]
        seed = self.seed_norm(self.station_seeds)[None, None].expand(b, g, -1, -1)
        scard = self.station_pool(seed, source, smask)
        svalid = smask.any(-1, keepdim=True).expand(-1, -1, self.station_tokens)
        return torch.cat((card, scard), 2), torch.cat((card_valid, svalid), 2)


class InteractionRound(nn.Module):
    def __init__(self, dim, heads, dropout, subgroups, station_tokens, globals_count):
        super().__init__()
        self.regional = RegionalSummaries(dim, heads, dropout, subgroups, station_tokens)
        self.global_seeds = nn.Parameter(torch.randn(globals_count, dim) / math.sqrt(dim))
        self.global_read = CrossUpdate(dim, heads, dropout)
        self.regional_read = CrossUpdate(dim, heads, dropout)
        self.grid_read = CrossUpdate(dim, heads, dropout)
        self.station_norm, self.grid_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.station_attention = Attention(dim, heads, dropout)
        self.grid_attention = Attention(dim, heads, dropout)
        self.station_fuse = nn.Linear(2*dim, dim, bias=False)
        self.station_drop, self.station_ffn = nn.Dropout(dropout), FeedForward(dim, dropout)

    def forward(self, station, grid, grid_valid, geometry, details=False):
        regional, regional_valid = self.regional(station, grid, grid_valid, geometry)
        b, _, _, d = regional.shape
        global_state = self.global_seeds[None].expand(b, -1, -1)
        global_state = self.global_read(global_state, regional.flatten(1, 2), regional_valid.flatten(1, 2))
        global_valid = regional_valid.flatten(1, 2).any(-1, keepdim=True).expand(-1, len(self.global_seeds))
        regional = self.regional_read(regional.flatten(1, 2), global_state, global_valid,
                                      regional_valid.flatten(1, 2)).reshape_as(regional)

        # Write regional messages to grids FIRST. Project region K/V once.
        keys, values = self.grid_read.attention.project_kv(self.grid_read.kv_norm(regional))
        keys, values = keys[:, geometry.block_region], values[:, geometry.block_region]
        rvalid = regional_valid[:, geometry.block_region]
        message = self.grid_read.attention.attend(self.grid_read.q_norm(grid), keys, values, rvalid)
        new_grid = self.grid_read.ffn(grid + self.grid_read.drop(message))
        new_grid = new_grid * grid_valid[..., None].to(new_grid.dtype)

        # All stations read OLD station states and the SAME newly updated grids.
        # There is deliberately no regional_read(...station...) branch.
        s = self.station_norm(station)
        keys, values = self.station_attention.project_kv(s)
        keys, values = keys[:, geometry.station_neighbors], values[:, geometry.station_neighbors]
        smask = geometry.station_neighbor_valid[None].expand(b, -1, -1)
        station_message = self.station_attention.attend(s[:, :, None], keys, values, smask).squeeze(-2)
        keys, values = self.grid_attention.project_kv(self.grid_norm(new_grid))
        keys = keys[:, geometry.candidate_to_unique].flatten(2, 3)
        values = values[:, geometry.candidate_to_unique].flatten(2, 3)
        gmask = (grid_valid[:, geometry.candidate_to_unique] &
                 geometry.candidate_valid[None, :, :, None]).flatten(2, 3)
        grid_message = self.grid_attention.attend(s[:, :, None], keys, values, gmask).squeeze(-2)
        new_station = station + self.station_drop(self.station_fuse(torch.cat((station_message, grid_message), -1)))
        new_station = self.station_ffn(new_station)
        detail = dict(station=new_station, grid=new_grid, regional=regional,
                      global_state=global_state) if details else None
        return new_station, new_grid, detail


class GeometryBuffers(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        for name, value in arrays.items():
            tensor = torch.as_tensor(np.asarray(value).copy())
            self.register_buffer(name, tensor)


class HierarchicalInteraction(nn.Module):
    def __init__(self, geometry, hidden=32, heads=4, rounds=4, subgroups=8,
                 station_tokens=2, globals_count=16, dropout=0.1,
                 patch_size=5, chunk_blocks=8192):
        super().__init__()
        if min(hidden, heads, rounds, subgroups, station_tokens, globals_count, chunk_blocks) < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid dimensions/dropout.')
        self.hidden, self.patch_size = hidden, patch_size
        self.geometry = GeometryBuffers(geometry)
        self.nodes = len(geometry['station_region'])
        self.blocks = len(geometry['block_region'])
        self.station_encoder = StationEncoder(hidden, dropout)
        self.grid_encoder = GridEncoder(hidden, patch_size, chunk_blocks)
        self.interactions = nn.ModuleList(InteractionRound(hidden, heads, dropout, subgroups,
                                                          station_tokens, globals_count) for _ in range(rounds))
        self.prediction_head = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, 12))

    def forward(self, station_input, patches, block_valid, return_details=False):
        if station_input.shape[1:] != (2, self.nodes, 12):
            raise ValueError('Station input must be (B,2,N,12).')
        if patches.shape != (len(station_input), self.blocks, 12, 2, self.patch_size, self.patch_size):
            raise ValueError('Grid input must be (B,U,12,2,P,P), in frozen unique-block order.')
        if block_valid.shape != patches.shape[:2] or block_valid.dtype != torch.bool:
            raise ValueError('Invalid grid block mask.')
        station = self.station_encoder(station_input)
        grid, valid = self.grid_encoder(patches, block_valid)
        trace = []
        for block in self.interactions:
            station, grid, detail = block(station, grid, valid, self.geometry, return_details)
            if return_details:
                trace.append(detail)
        prediction = self.prediction_head(station).transpose(1, 2).contiguous()
        if return_details:
            return prediction, dict(rounds=trace, station=station, grid=grid, grid_valid=valid)
        return prediction
