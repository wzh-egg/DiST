"""Observed-event MAE, deterministic AMP retries, evaluation and saved state."""
import csv
from contextlib import nullcontext
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.loss import relative_objective
from src.utils import Errors, batches
from src.timing import PhaseTimer
from src.input_pipeline import Pipeline, assemble


def capture_rng(rng=None, cuda=False):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if cuda else [],
                shuffle=rng.bit_generator.state if rng is not None else None)


def restore_rng(state, rng=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    if rng is not None:
        rng.bit_generator.state = state['shuffle']


def host_tensors(host, device):
    return assemble(**{k: torch.from_numpy(v).to(device) for k, v in host.arrays.items()})


class UnprofiledUpdate:
    """Keep pipeline synchronization in Pipeline; skip optional timing events."""
    source_read = 0.

    def phase(self, name, gpu=True):
        return nullcontext()

    def result(self):
        return None


def train_update(model, optimizer, source, origins, micro_batch, device, scaler,
                 pipeline=None, clip=5.0, max_retries=8, profile_update=True, relative_weight=0., relative_floor=1.):
    """One Adam update, globally weighted over all observed effective-batch labels.

    The prefetch plan is consumed only on the first attempt. On overflow, reread
    the same frozen origins from the immutable cache and replay the whole update
    with restored dropout RNG. No samples are dropped and no pinned slots are
    retained beyond backward (microbatch accumulation supports batch32/4 etc.).
    """
    data, device = source.station, torch.device(device)
    total = sum(int(data.observed[int(t):int(t)+data.P].sum()) for t in origins)
    if total == 0:
        raise ValueError('Effective batch has no observed labels.')
    if relative_weight < 0 or relative_floor <= 0:
        raise ValueError('Invalid relative loss parameters')
    nonzero_total = sum(int((data.observed[int(t):int(t)+data.P] &
        (data.targets[int(t):int(t)+data.P] != 0)).sum()) for t in origins) if relative_weight else 0
    state = capture_rng(cuda=device.type == 'cuda')
    timer = PhaseTimer(device) if profile_update else UnprofiledUpdate()
    for attempt in range(max_retries + 1):
        if attempt:
            restore_rng(state)
        optimizer.zero_grad(set_to_none=True)
        errors, finite, seen = [], [], 0
        objectives = []
        for group in batches(origins, micro_batch):
            packet = None
            try:
                with timer.phase('data_wait', gpu=False):
                    if pipeline is not None and attempt == 0:
                        packet = pipeline.take(group)
                        host = packet.host
                    else:
                        host = source.read(group)
                timer.source_read += host.read_seconds
                if not host.observed.any():
                    continue
                with timer.phase('input_h2d_assembly'):
                    inputs = packet.tensors(device) if packet is not None else host_tensors(host, device)
                    labels = torch.from_numpy(host.target).to(device)
                    mask = torch.from_numpy(host.observed).to(device)
                with timer.phase('forward'):
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                        output = model(*inputs)
                with timer.phase('loss'):
                    raw = output.float() * data.std + data.mean
                    if raw.shape != labels.shape:
                        raise ValueError('Prediction/target dimensions differ.')
                    finite.append(torch.isfinite(raw).all())
                    error = (raw[mask] - labels[mask]).abs()
                    loss = relative_objective(error, labels[mask], total, nonzero_total, relative_weight, relative_floor)
                with timer.phase('backward'):
                    scaler.scale(loss).backward()
                errors.append(error.detach().double().sum())
                objectives.append(loss.detach().double())
                seen += error.numel()
                del output, raw, error, loss, inputs, labels, mask
            finally:
                if packet is not None:
                    pipeline.release()
        if seen != total:
            raise ValueError('Effective-batch observed label count mismatch.')
        with timer.phase('gradient_check_clip'):
            scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    raise ValueError('Missing gradient: ' + name)
            # clip_grad_norm_ already computes norms of ALL gradients. A finite
            # total norm implies all gradients are finite; avoid one isfinite
            # reduction kernel per Parameter. Invalid clipped gradients are
            # discarded by the same clear/replay path, never passed to Adam.
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip, foreach=True)
            finite.append(torch.isfinite(norm))
            good = bool(torch.stack(finite).all())
        if good:
            with timer.phase('optimizer'):
                scaler.step(optimizer)
                scaler.update()
            absolute = float(torch.stack(errors).sum())
            return dict(absolute=absolute, count=seen, objective=float(torch.stack(objectives).sum()), retries=attempt, timing=timer.result())
        if not scaler.is_enabled():
            raise ValueError('Nonfinite FP32 output/gradient; optimizer not updated.')
        # Explicitly skip Adam, clear scaler per-optimizer state, reduce scale.
        scaler.update(new_scale=scaler.get_scale() / 2.0)
    raise RuntimeError('AMP overflow persisted after retries; no batch was dropped.')


@torch.no_grad()
def evaluate(model, source, origins, size, device):
    model.eval()
    device = torch.device(device)
    data = source.station
    metrics = dict(model=Errors(data.P), persistence=Errors(data.P))
    groups = list(batches(origins, size))
    started = time.perf_counter()
    timer = PhaseTimer(device)


    def consume(host, inputs):
        with timer.phase('forward'):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                prediction = model(*inputs)
        with timer.phase('metrics_and_output_transfer', gpu=False):
            raw = (prediction.float() * data.std + data.mean).cpu().numpy()
            metrics['model'].add(raw, host.target, host.observed)
            metrics['persistence'].add(host.last[:, None], host.target, host.observed)

    if device.type == 'cuda':
        with Pipeline(source, groups, 'prefetch', device) as pipe:
            for number, group in enumerate(groups, 1):
                with timer.phase('data_wait', gpu=False):
                    packet = pipe.take(group)
                timer.source_read += packet.host.read_seconds
                try:
                    with timer.phase('input_h2d_assembly'):
                        inputs = packet.tensors(device)
                    consume(packet.host, inputs)
                    del inputs
                finally:
                    pipe.release()
                if number % 100 == 0:
                    print('Validation {}/{}'.format(number, len(groups)), flush=True)
            if pipe.position != len(groups) or pipe.stats()['windows'] != len(origins):
                raise ValueError('Incomplete evaluation coverage.')
    else:
        for group in groups:
            with timer.phase('data_wait', gpu=False):
                host = source.read(group)
            timer.source_read += host.read_seconds
            with timer.phase('input_h2d_assembly'):
                inputs = host_tensors(host, device)
            consume(host, inputs)
    result = dict(windows=len(origins), seconds=time.perf_counter()-started,
                  **{key: accumulator.result() for key, accumulator in metrics.items()})
    result['timing'] = timer.result()
    if result['model']['count'] != result['persistence']['count']:
        raise ValueError('Paired metric counts differ.')
    return result


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
