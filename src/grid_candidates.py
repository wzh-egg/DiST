"""Restrict frozen read indices while preserving the cell partition and grid bank."""
import numpy as np


def restrict_candidates(arrays, count):
    mask = arrays["candidate_valid"]
    if mask.ndim != 2 or mask.shape[1] != 24 or count not in (4, 8, 16, 24):
        raise ValueError("Expected the original 24 ring×direction slots")
    if count == 24:
        return arrays, np.broadcast_to(np.arange(24), mask.shape).copy()
    slots = np.empty((len(mask), count), dtype=np.int64)
    for station in range(len(mask)):
        per_direction = []
        for direction in range(8):
            start = (station + direction) % 3
            choices = [((start + offset) % 3) * 8 + direction for offset in range(3)]
            choices.sort(key=lambda slot: not mask[station, slot])
            per_direction.append(choices)
        if count == 4:
            for pair in range(4):
                first = 2 * pair + (station + pair) % 2
                two = [per_direction[first][0], per_direction[first ^ 1][0]]
                slots[station, pair] = next((slot for slot in two if mask[station, slot]), two[0])
        else:
            slots[station] = [choices[layer] for layer in range(count // 8)
                              for choices in per_direction]
    valid = np.take_along_axis(mask, slots, axis=1)
    if not np.array_equal(valid.any(1), mask.any(1)):
        raise ValueError("Candidate selection lost all valid slots for a station")
    return dict(arrays, candidate_to_unique=np.take_along_axis(
        arrays["candidate_to_unique"], slots, axis=1), candidate_valid=valid), slots


