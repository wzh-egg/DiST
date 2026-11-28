"""Shared I/O and observed-event metrics. No torch dependency."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def batches(origins, size):
    for start in range(0, len(origins), size):
        yield origins[start:start + size]


class Errors:
    """Float64 sums over valid forecast events, MAPE on targets > 0 only."""
    def __init__(self, horizon):
        self.count = np.zeros(horizon, np.int64)
        self.absolute = np.zeros(horizon, np.float64)
        self.squared = np.zeros(horizon, np.float64)
        self.mape_count = np.zeros(horizon, np.int64)
        self.percentage = np.zeros(horizon, np.float64)

    def add(self, prediction, target, mask):
        prediction = np.broadcast_to(prediction, target.shape)
        if target.ndim != 3 or target.shape[1] != len(self.count):
            raise ValueError('Metrics require (batch,horizon,nodes).')
        if mask.shape != target.shape or mask.dtype != bool:
            raise ValueError('Invalid label mask.')
        if not np.array_equal(mask, np.isfinite(target)):
            raise ValueError('Mask must represent finite, unfilled station labels.')
        if not np.isfinite(prediction).all():
            raise ValueError('Non-finite prediction.')
        error = np.zeros(target.shape, np.float64)
        np.subtract(prediction, target, out=error, where=mask)
        absolute = np.abs(error)
        self.count += mask.sum(axis=(0, 2), dtype=np.int64)
        self.absolute += absolute.sum(axis=(0, 2), dtype=np.float64)
        self.squared += np.square(error).sum(axis=(0, 2), dtype=np.float64)
        positive = mask & (target > 0)
        percentage = np.zeros(target.shape, np.float64)
        np.divide(absolute, target, out=percentage, where=positive)
        self.mape_count += positive.sum(axis=(0, 2), dtype=np.int64)
        self.percentage += percentage.sum(axis=(0, 2), dtype=np.float64) * 100

    def result(self):
        def aggregate(indices):
            n = int(self.count[indices].sum())
            m = int(self.mape_count[indices].sum())
            return dict(count=n, mae=float(self.absolute[indices].sum() / n) if n else None,
                        rmse=math.sqrt(float(self.squared[indices].sum() / n)) if n else None,
                        mape_percent=float(self.percentage[indices].sum() / m) if m else None,
                        mape_count=m)
        if not self.count.sum():
            raise ValueError('No observed labels.')
        result = aggregate(slice(None))
        result['per_hour'] = [dict(hour=h + 1, **aggregate(slice(h, h + 1)))
                              for h in range(len(self.count))]
        return result
