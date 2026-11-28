"""Existing AQI preparation reader, unchanged class body."""
import json
from pathlib import Path
import numpy as np

class Prepared:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / 'READY').is_file():
            raise ValueError('Prepared directory has no READY completion marker.')
        self.report = json.loads((self.root / 'report.json').read_text())
        for name in ('features', 'targets', 'observed', 'persistence',
                     'station_ids', 'timestamps'):
            setattr(self, name, np.load(self.root / (name + '.npy'),
                                       mmap_mode='r', allow_pickle=False))
        self.T, self.N = self.targets.shape
        self.H = int(self.report['history'])
        self.P = int(self.report['horizon'])
        self.mean = float(self.report['train_mean'])
        self.std = float(self.report['train_std'])
        if self.H != 12 or self.P != 12:
            raise ValueError('This GWN configuration requires 12 -> 12 hours.')
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError('Invalid training normalization.')
        if self.features.shape != (self.T, self.N, 2):
            raise ValueError('Input features must have shape (T,N,2).')
        if self.observed.shape != self.targets.shape or self.observed.dtype != bool:
            raise ValueError('Invalid observed label mask.')
        if self.persistence.shape != self.targets.shape:
            raise ValueError('Invalid persistence shape.')
        if self.station_ids.shape != (self.N,) or len(set(self.station_ids.tolist())) != self.N:
            raise ValueError('Station IDs must be unique and aligned.')
        if self.timestamps.shape != (self.T,) or not np.all(
                np.diff(self.timestamps.astype('datetime64[h]')) == np.timedelta64(1, 'h')):
            raise ValueError('Hourly timeline must be continuous.')
        if self.T != self.report['hours'] or self.N != self.report['selected_stations']:
            raise ValueError('Shape does not match preparation report.')
        for start in range(0, self.T, 512):
            s = slice(start, start + 512)
            if not np.isfinite(self.features[s]).all() or not np.isfinite(self.persistence[s]).all():
                raise ValueError('Non-finite prepared inputs or persistence.')
            if not np.array_equal(self.observed[s], np.isfinite(self.targets[s])):
                raise ValueError('Label mask disagrees with finite raw targets.')
        a, b = int(self.report['train_end_index']), int(self.report['val_end_index'])
        self.origins = {}
        for name, start, end in [('train', 0, a), ('val', a, b), ('test', b, self.T)]:
            x = np.load(self.root / (name + '_origins.npy'), allow_pickle=False)
            if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) or not len(x):
                raise ValueError(f'Invalid or empty {name} origins.')
            if not np.all(np.diff(x) > 0) or x[0] < max(self.H, start) or x[-1] + self.P > end:
                raise ValueError(f'{name} origins violate temporal boundaries.')
            if len(x) != self.report['splits'][name]['kept_windows']:
                raise ValueError(f'{name} window count disagrees with report.')
            self.origins[name] = x

    def batch(self, origins):
        x = np.stack([self.features[t-self.H:t].transpose(2, 1, 0) for t in origins])
        y = np.stack([self.targets[t:t+self.P] for t in origins])
        mask = np.stack([self.observed[t:t+self.P] for t in origins])
        baseline = np.stack([self.persistence[t-1] for t in origins])
        return np.ascontiguousarray(x, dtype=np.float32), y, mask, baseline
