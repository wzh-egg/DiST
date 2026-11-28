"""Phase timing and epoch-validation early stopping with exact update indices."""
from collections import defaultdict
from contextlib import contextmanager
import math
import time

import torch


class PhaseTimer:
    """CPU wall times plus CUDA event intervals on the consumer stream.

    CUDA intervals include stream waits/idle/host launch gaps, not pure kernel
    time. Background reader and H2D times overlap compute and must not be added
    to the end-to-end wall time. Events are resolved once at the batch boundary.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.wall = defaultdict(float)
        self.events = []
        self.source_read = 0.0

    @contextmanager
    def phase(self, name, gpu=True):
        start = time.perf_counter()
        pair = None
        if gpu and self.device.type == 'cuda':
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record(torch.cuda.current_stream(self.device))
                self.events.append((name, pair))
            self.wall[name] += time.perf_counter() - start

    def result(self):
        cuda = defaultdict(float)
        if self.events:
            self.events[-1][1][1].synchronize()
            for name, (start, end) in self.events:
                cuda[name] += start.elapsed_time(end)/1000.0
        return dict(wall_seconds=time.perf_counter()-self.started,
                    phase_wall_seconds=dict(self.wall), phase_cuda_interval_seconds=dict(cuda),
                    source_read_cpu_seconds=self.source_read)


class EarlyStopping:
    """Strict validation-MAE improvement, checked after each COMPLETE epoch.

    An update index counts successful effective-batch optimizer updates, not
    microbatches or AMP retry attempts. The best batch is the last update before
    the winning complete validation, not a claim that every batch was validated.
    """
    def __init__(self, patience, state=None):
        if patience < 1:
            raise ValueError('Patience must be positive.')
        self.patience = patience
        self.best = float('inf')
        self.best_epoch = self.best_global_update = self.best_update_in_epoch = 0
        self.stale = self.last_epoch = self.last_global_update = self.last_update_in_epoch = 0
        if state is not None:
            if state['patience'] != patience:
                raise ValueError('Early-stop patience changed.')
            for name in ('best', 'best_epoch', 'best_global_update', 'best_update_in_epoch',
                         'stale', 'last_epoch', 'last_global_update', 'last_update_in_epoch'):
                setattr(self, name, state[name])

    @property
    def should_stop(self):
        return self.stale >= self.patience

    def observe(self, value, epoch, global_update, update_in_epoch):
        if not math.isfinite(value):
            raise ValueError('Early stop requires finite validation MAE.')
        if epoch <= self.last_epoch or global_update <= self.last_global_update or update_in_epoch < 1:
            raise ValueError('Validation epoch/update indices must increase.')
        improved = value < self.best
        if improved:
            self.best, self.best_epoch = value, epoch
            self.best_global_update, self.best_update_in_epoch = global_update, update_in_epoch
            self.stale = 0
        else:
            self.stale += 1
        self.last_epoch, self.last_global_update, self.last_update_in_epoch = epoch, global_update, update_in_epoch
        return improved

    def state(self, reason='continue'):
        return dict(patience=self.patience, best=self.best, best_epoch=self.best_epoch,
                    best_global_update=self.best_global_update, best_update_in_epoch=self.best_update_in_epoch,
                    stale=self.stale, last_epoch=self.last_epoch, last_global_update=self.last_global_update,
                    last_update_in_epoch=self.last_update_in_epoch, stopped_early=self.should_stop,
                    stop_reason=reason, check_scope='complete validation epoch; no per-batch validation',
                    update_unit='successful effective-batch optimizer step; retries/microbatches excluded')


def add_timing(total, row):
    total['wall_seconds'] += row['wall_seconds']
    total['source_read_cpu_seconds'] += row['source_read_cpu_seconds']
    for kind in ('phase_wall_seconds', 'phase_cuda_interval_seconds'):
        for key, value in row[kind].items():
            total[kind][key] = total[kind].get(key, 0.0) + value


def empty_timing():
    return dict(wall_seconds=0.0, source_read_cpu_seconds=0.0,
                phase_wall_seconds={}, phase_cuda_interval_seconds={})
