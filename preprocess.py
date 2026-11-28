"""Construct AIR's fixed region partition and station/grid read indices."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from src.region_geometry import build_geometry
from src.region_neighbors import enrich
from src.grid_candidates import restrict_candidates


def prepare(coordinates, candidates, output):
    xy = np.load(coordinates, allow_pickle=False)
    with np.load(candidates, allow_pickle=False) as archive:
        blocks = archive["center_lon_lat"][archive["unique_ids"]]
        arrays, report = build_geometry(
            xy, blocks, archive["candidate_to_unique"], archive["candidate_valid"],
            regions=32, subgroups=8, station_neighbors=8, station_tokens=2,
        )
    arrays, _ = restrict_candidates(enrich(arrays, report, xy), 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.coordinates, args.candidates, args.output)
