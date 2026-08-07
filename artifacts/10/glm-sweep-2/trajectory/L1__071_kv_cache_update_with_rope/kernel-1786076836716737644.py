import torch


def _rope_apply(key_states, cos, sin):
    head_dim = key_states.shape[-1]
    half_dim = head_dim // 2
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    return (key_states * cos) + (k_rotated * sin)


_rope_compiled = torch.compile(_rope_apply, mode="reduce-overhead")


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Fused KV cache update with RoPE application.
    """
    key_states_rotated = _rope_compiled(key_states, cos, sin)

    updated_key_cache = torch.cat([key_cache, key_states_rotated], dim=2)
    updated_value_cache = torch.cat([value_cache, value_states], dim=2)

    return updated_key_cache, updated_value_cache
