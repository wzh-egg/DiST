"""Read the hourly AIR station series and its time-aligned grid patch bank."""
from types import SimpleNamespace
import numpy as np
from src.station_data import Prepared
from src.input_pipeline import Source


def source_from(station_dir, grid):
    station = Prepared(station_dir)
    values = np.load(grid / "normalized.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(grid / "valid.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != mask.shape or values.ndim != 4 or values.shape[0] != station.T or values.shape[-2:] != (5, 5):
        raise ValueError("Grid values and masks must align with station hours and 5-by-5 patches")
    if values.dtype != np.float32 or mask.dtype != bool:
        raise ValueError("Grid patches require float32 values and a boolean validity mask")
    return Source(station, SimpleNamespace(values=values, valid=mask))
