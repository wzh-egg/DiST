"""Keep frozen leaf cells; reconstruct station membership and touching neighbors."""
import numpy as np


def enrich(arrays, report, station_xy):
    out = dict(arrays)
    count = len(arrays['group_ids'])
    depth = int(np.log2(count))
    planes = {(p['depth'], p['node']): (p['axis'], p['cut']) for p in report['split_planes']}
    bounds = np.zeros((count, 4), np.float64)
    membership = np.zeros(len(station_xy), np.int64)
    for level in range(depth):
        previous = membership.copy()
        for prefix in np.unique(previous):
            axis, cut = planes[level, int(prefix)]
            take = np.flatnonzero(previous == prefix)
            membership[take] = 2*prefix + (station_xy[take, axis] > cut)
    def visit(level, prefix, box):
        if level == depth:
            bounds[prefix] = box
            return
        axis, cut = planes[level, prefix]
        left, right = box.copy(), box.copy()
        left[axis+2] = cut; right[axis] = cut
        visit(level+1, 2*prefix, left);visit(level+1, 2*prefix+1, right)
    visit(0, 0, np.array([-180., -90., 180., 90.]))
    # Closed cells touch on an edge or corner. Include own cell separately.
    overlap = np.minimum(bounds[:,None,2:], bounds[None,:,2:]) - np.maximum(bounds[:,None,:2], bounds[None,:,:2])
    adjacent = (overlap >= -1e-10).all(-1)
    width = max(1, int(np.bincount(membership,minlength=count).max()))
    ids = np.zeros((count,width),np.int64);valid=np.zeros_like(ids,bool)
    for cell in range(count):
        take=np.flatnonzero(membership==cell);ids[cell,:len(take)]=take;valid[cell,:len(take)]=True
    subgroups=report['subgroups_per_region'];old_slots=arrays['card_coordinates_lon_lat'].shape[0]//report['regions']
    card_ids=np.array([i//subgroups*old_slots+i%subgroups for i in range(count)])
    out.update(cell_station_ids=ids,cell_station_valid=valid,station_cell=membership,
               cell_adjacent=adjacent,cell_bounds=bounds,
               cell_coordinates=arrays['card_coordinates_lon_lat'][card_ids],
               cell_relative=arrays['card_relative_features'][card_ids[:,None],card_ids[None,:]])
    return out
