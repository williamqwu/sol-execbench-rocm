import torch


def _fused_run(key_states, value_states, cos, sin, key_cache, value_cache):
    B, H, S_old, D = key_cache.shape
    S_new = key_states.shape[2]
    half = D // 2

    out_key = torch.empty(B, H, S_old + S_new, D, dtype=key_cache.dtype, device=key_cache.device)
    out_val = torch.empty(B, H, S_old + S_new, D, dtype=value_cache.dtype, device=value_cache.device)

    out_key[:, :, :S_old] = key_cache
    out_val[:, :, :S_old] = value_cache

    k1 = key_states[..., :half]
    k2 = key_states[..., half:]
    c1 = cos[..., :half]
    c2 = cos[..., half:]
    s1 = sin[..., :half]
    s2 = sin[..., half:]

    out_key[:, :, S_old:, :half] = k1 * c1 - k2 * s1
    out_key[:, :, S_old:, half:] = k2 * c2 + k1 * s2
    out_val[:, :, S_old:] = value_states

    return out_key, out_val


_compiled = torch.compile(_fused_run)


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    return _compiled(key_states, value_states, cos, sin, key_cache, value_cache)
