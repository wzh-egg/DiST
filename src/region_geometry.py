"""Value-free card positions and relative geometry; original grouping unchanged."""
import numpy as np
from src.geometry import build_geometry as original_build_geometry

EXTRA_KEYS = ('card_coordinates_lon_lat', 'card_relative_features')


def centroid(xy):
    """Spherical centroid of the static source locations, not learned pool weights."""
    radians = np.deg2rad(np.asarray(xy, np.float64))
    lon, lat = radians.T
    xyz = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), -1).mean(0)
    if np.linalg.norm(xyz) < 1e-12:
        raise ValueError('Card sources have an undefined spherical centroid.')
    return np.rad2deg([np.arctan2(xyz[1], xyz[0]), np.arctan2(xyz[2], np.hypot(*xyz[:2]))])


def relative_features(coordinates):
    """[east/1000km, north/1000km, log1p(great-circle km /100)] from query to source.

    Signed offsets use a local equirectangular approximation. The radial feature
    uses haversine distance. Coordinates are geographic metadata only.
    """
    xy = np.deg2rad(np.asarray(coordinates, np.float64))
    delta = xy[None, :, :] - xy[:, None, :]
    dlon = (delta[..., 0] + np.pi) % (2*np.pi) - np.pi
    dlat = delta[..., 1]
    mean_lat = (xy[None, :, 1] + xy[:, None, 1])/2
    east = 6371.0088 * dlon * np.cos(mean_lat)
    north = 6371.0088 * dlat
    a = np.sin(dlat/2)**2 + np.cos(xy[:, None, 1])*np.cos(xy[None, :, 1])*np.sin(dlon/2)**2
    distance = 2*6371.0088*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    result = np.stack((east/1000, north/1000, np.log1p(distance/100)), -1)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite card geometry.')
    return result.astype(np.float32)


def add_card_geometry(arrays, station_xy, block_xy, subgroups=8, station_tokens=2):
    if any(key in arrays for key in EXTRA_KEYS):
        raise ValueError('Card geometry already present.')
    result = dict(arrays)
    regions = len(arrays['region_station_ids'])
    positions = []
    for region in range(regions):
        for subgroup in range(subgroups):
            group = region*subgroups + subgroup
            ids = arrays['group_ids'][group, arrays['group_valid'][group]]
            positions.append(centroid(block_xy[ids]))
        ids = arrays['region_station_ids'][region, arrays['region_station_valid'][region]]
        # Empty station cards are masked by RegionalSummaries. Their static
        # fallback coordinate must be finite but is never a valid key/value.
        center = centroid(station_xy[ids]) if len(ids) else centroid(block_xy[arrays['block_region']==region])
        positions.extend([center.copy() for _ in range(station_tokens)])
    result['card_coordinates_lon_lat'] = np.asarray(positions, np.float32)
    result['card_relative_features'] = relative_features(result['card_coordinates_lon_lat'])
    return result


def build_geometry(station_xy, block_xy, candidate_ids, candidate_valid,
                   regions=32, subgroups=8, station_neighbors=8, station_tokens=2):
    arrays, report = original_build_geometry(station_xy, block_xy, candidate_ids,
        candidate_valid, regions, subgroups, station_neighbors)
    arrays = add_card_geometry(arrays, np.asarray(station_xy), np.asarray(block_xy), subgroups, station_tokens)
    report.update(card_count=regions*(subgroups+station_tokens),
        card_position='grid card: spherical centroid of subgroup blocks; both station cards: spherical centroid of region stations; static approximate source coverage',
        card_relative_features='source minus query: east/1000km, north/1000km, log1p(haversine_km/100)',
        spatial_sources='coordinates only; no AQI, forecast targets or split statistics')
    return arrays, report
