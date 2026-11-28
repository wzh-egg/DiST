"""Protect regional Q/K/V/O, norms, GeoMLP, and trainable mixture gates."""
import torch


def core_of(model):
    model = getattr(model, '_orig_mod', model)
    return getattr(model, 'core', model)


def protected_names(model):
    core = core_of(model)
    prefix = 'core.' if core is not model else ''
    names = set()
    for r, block in enumerate(core.interactions):
        start = f'{prefix}interactions.{r}.'
        names.update(start + 'regional_read.' + suffix for suffix in (
            'q_norm.weight', 'q_norm.bias', 'kv_norm.weight', 'kv_norm.bias',
            'attention.query.weight', 'attention.key.weight',
            'attention.value.weight', 'attention.output.weight'))
        names.update(start + 'scorer.' + n for n, _ in block.scorer.named_parameters())
    if hasattr(core, 'horizon_logits'):
        name = prefix + 'horizon_logits'
        names.add(name)
    return names


def make_optimizer(model, job):
    names = protected_names(model)
    named = dict(model.named_parameters())
    return torch.optim.Adam([
        dict(params=[v for n, v in named.items() if n not in names], weight_decay=job['weight_decay']),
        dict(params=[v for n, v in named.items() if n in names], weight_decay=0.)
    ], lr=job['lr'])
