"""Prepare hourly station inputs from the AIR observation matrix and split indices."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


def prepare(raw, split_dir, out):
    values = np.load(raw / "targets.npy", mmap_mode="r", allow_pickle=False)
    times = np.load(raw / "timestamps.npy", allow_pickle=False).astype("datetime64[h]")
    ids = np.load(raw / "station_ids.npy", allow_pickle=False)
    if values.shape != (len(times), len(ids)) or not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError("Expected a continuous hourly station matrix in chronological order")
    train_end, val_end = int(len(times)*0.7), int(len(times)*0.8)
    missing = (~np.isfinite(values[:train_end])).mean(0)
    selected = np.flatnonzero(missing <= 0.1)
    if not len(selected):
        raise ValueError("No station meets the training coverage threshold")
    y = np.ascontiguousarray(values[:, selected], dtype=np.float32)
    observed = np.isfinite(y)
    fit = y[:train_end][observed[:train_end]].astype(np.float64)
    mean, std = float(fit.mean()), float(fit.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training observations must have positive variance")
    splits = {}
    arrays = {}
    for name, lo, hi in (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, len(times))):
        origins = np.load(split_dir / (name + "_origins.npy"), allow_pickle=False)
        if (origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or not origins.size
                or origins[0] < max(12, lo) or origins[-1]+12 > hi or np.any(np.diff(origins) <= 0)):
            raise ValueError("Invalid forecast origins for " + name)
        arrays[name+"_origins"] = origins
        splits[name] = {"kept_windows":len(origins)}
    features = np.empty((*y.shape, 2), np.float32)
    persistence = np.empty_like(y)
    last = np.full(len(selected), mean, np.float32)
    last_seen = np.full(len(selected), -4, np.int64)
    for t, valid in enumerate(observed):
        last[valid] = y[t, valid]
        last_seen[valid] = t
        persistence[t] = last
        features[t, :, 0] = np.where(t-last_seen <= 3, (last-mean)/std, 0.)
    features[:, :, 1] = ((times.astype(np.int64) % 24)/24)[:, None]
    arrays.update(features=features, targets=y, observed=observed, persistence=persistence,
                  timestamps=times, station_ids=ids[selected], station_indices=selected,
                  training_missing=missing[selected])
    report = dict(history=12, horizon=12, train_mean=mean, train_std=std, hours=len(times),
                  selected_stations=len(selected), train_end_index=train_end, val_end_index=val_end, splits=splits)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        prepared = Path(tmp)/"stations"
        prepared.mkdir()
        for key, value in arrays.items():
            np.save(prepared/(key+".npy"), value, allow_pickle=False)
        (prepared/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        (prepared/"READY").touch()
        prepared.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("raw", "split_dir", "out"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.split_dir, args.out)
