"""Frozen, value-free spatial groups and independent nearest-station links."""
import math
from pathlib import Path

import numpy as np


def _power_two(value):
    return value > 0 and value & (value - 1) == 0


def _packed(labels, count):
    groups = [np.flatnonzero(labels == i) for i in range(count)]
    width = max(1, max(map(len, groups)))
    ids = np.zeros((count, width), np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    for i, group in enumerate(groups):
        ids[i, :len(group)] = group
        valid[i, :len(group)] = True
    return ids, valid


def _stats(counts):
    return dict(min=int(counts.min()), median=float(np.median(counts)),
                mean=float(counts.mean()), max=int(counts.max()))


def nearest_stations(coordinates, count):
    """Exact spherical distance, stable index tie-break, no grid-derived filter."""
    xy = np.deg2rad(coordinates)
    xyz = np.column_stack((np.cos(xy[:, 1]) * np.cos(xy[:, 0]),
                           np.cos(xy[:, 1]) * np.sin(xy[:, 0]), np.sin(xy[:, 1])))
    n = len(xyz)
    ids = np.zeros((n, count), np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    for i in range(n):
        distance = np.sum((xyz - xyz[i]) ** 2, axis=1)
        distance[i] = np.inf
        take = np.lexsort((np.arange(n), distance))[:min(count, n - 1)]
        ids[i, :len(take)] = take
        valid[i, :len(take)] = True
    return ids, valid


def build_geometry(station_xy, block_xy, candidate_ids, candidate_valid,
                   regions=32, subgroups=8, station_neighbors=8):
    """Split block centers by alternating longitude/latitude medians.

    First log2(regions) levels define regions; the remaining levels define
    subgroups. All unique input blocks belong to exactly one subgroup. Stations
    follow the saved geographic split planes, independently of grid candidates.
    No measured AQI, training labels or validation/test values are consulted.
    """
    station_xy = np.asarray(station_xy, np.float64)
    block_xy = np.asarray(block_xy, np.float64)
    candidate_ids = np.asarray(candidate_ids, np.int64)
    candidate_valid = np.asarray(candidate_valid, bool)
    if not _power_two(regions) or not _power_two(subgroups):
        raise ValueError('Regions and subgroups must be powers of two.')
    if station_neighbors < 1 or len(block_xy) < regions * subgroups:
        raise ValueError('Need at least one block per subgroup and positive station K.')
    for xy in (station_xy, block_xy):
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
            raise ValueError('Coordinates must be finite longitude/latitude pairs.')
        if (np.abs(xy) > [180, 90]).any():
            raise ValueError('Coordinates outside longitude/latitude range.')
    n, u = len(station_xy), len(block_xy)
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != n or candidate_ids.shape != candidate_valid.shape:
        raise ValueError('Grid candidates must be aligned (N,K) arrays.')
    if not candidate_valid.any() or (candidate_ids[candidate_valid] < 0).any() or (candidate_ids[candidate_valid] >= u).any():
        raise ValueError('Invalid grid candidate IDs.')
    region_depth = int(math.log2(regions))
    total_depth = region_depth + int(math.log2(subgroups))
    block_group = np.empty(u, np.int64)
    station_group = np.empty(n, np.int64)
    planes = []

    def split(blocks, stations, depth, prefix):
        if depth == total_depth:
            block_group[blocks] = prefix
            station_group[stations] = prefix
            return
        axis = depth % 2
        order = blocks[np.lexsort((blocks, block_xy[blocks, axis]))]
        middle = len(order) // 2
        cut = float((block_xy[order[middle-1], axis] + block_xy[order[middle], axis]) / 2)
        planes.append(dict(depth=depth, node=prefix, axis=axis, cut=cut))
        left = station_xy[stations, axis] <= cut
        split(order[:middle], stations[left], depth + 1, prefix * 2)
        split(order[middle:], stations[~left], depth + 1, prefix * 2 + 1)

    split(np.arange(u), np.arange(n), 0, 0)
    block_region = block_group // subgroups
    station_region = station_group // subgroups
    group_ids, group_valid = _packed(block_group, regions * subgroups)
    region_station_ids, region_station_valid = _packed(station_region, regions)
    neighbors, neighbor_valid = nearest_stations(station_xy, station_neighbors)
    arrays = dict(block_group=block_group, block_region=block_region,
                  station_region=station_region, group_ids=group_ids, group_valid=group_valid,
                  region_station_ids=region_station_ids, region_station_valid=region_station_valid,
                  station_neighbors=neighbors, station_neighbor_valid=neighbor_valid,
                  candidate_to_unique=np.where(candidate_valid, candidate_ids, 0),
                  candidate_valid=candidate_valid)
    report = dict(regions=regions, subgroups_per_region=subgroups, nodes=n, unique_blocks=u,
                  blocks_per_region=_stats(np.bincount(block_region, minlength=regions)),
                  blocks_per_subgroup=_stats(group_valid.sum(1)),
                  stations_per_region=_stats(region_station_valid.sum(1)),
                  empty_station_regions=int((~region_station_valid.any(1)).sum()),
                  station_neighbor_slots=station_neighbors, grid_candidate_slots=candidate_ids.shape[1],
                  station_selection='independent exact nearest real stations; exclude self; no grid restriction',
                  partition='alternating longitude/latitude block-count median split; fixed tie by block index',
                  split_planes=planes,
                  scope='all unique blocks supplied by the existing cache, not the complete national grid')
    return arrays, report


def load_geometry(path):
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}
