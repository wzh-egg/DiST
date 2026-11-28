"""Observed MAE plus capped relative error, in original AQI units."""


def relative_objective(error, truth, total, nonzero_total, weight, floor):
    loss = error.sum() / total
    if weight and nonzero_total:
        nonzero = truth != 0
        relative = (error[nonzero] / truth[nonzero].abs().clamp_min(floor)).sum()
        loss = loss + weight * floor * relative / nonzero_total
    return loss
