import torch


@torch.compile(dynamic=True)
def _compiled(key_states, value_states, cos, sin, key_cache, value_cache):
    k1 = key_states[..., :64]
    k2 = key_states[..., 64:]
    r1 = k1 * cos[..., :64] - k2 * sin[..., :64]
    r2 = k2 * cos[..., 64:] + k1 * sin[..., 64:]
    rotated = torch.cat((r1, r2), dim=-1)
    return (torch.cat((key_cache, rotated), dim=2),
            torch.cat((value_cache, value_states), dim=2))


@torch.no_grad()
def run(key_states, value_states, cos, sin, key_cache, value_cache):
    return _compiled(key_states, value_states, cos, sin, key_cache, value_cache)
