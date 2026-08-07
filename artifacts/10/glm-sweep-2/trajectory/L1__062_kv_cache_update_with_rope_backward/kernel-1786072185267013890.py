import torch

@torch.no_grad()
def _run_impl(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    half_dim = key_states.shape[-1] // 2

    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    grad_key_states_rotated = grad_key_cache[:, :, cache_position]
    grad_value_states = grad_value_cache[:, :, cache_position]

    grad_from_cos_term = grad_key_states_rotated * cos_expanded
    grad_k_rotated_half = grad_key_states_rotated * sin_expanded

    grad_k_rotated_half_1 = grad_k_rotated_half[..., :half_dim]
    grad_k_rotated_half_2 = grad_k_rotated_half[..., half_dim:]

    grad_k2_from_rotate = -grad_k_rotated_half_1
    grad_k1_from_rotate = grad_k_rotated_half_2

    grad_k1_total = grad_from_cos_term[..., :half_dim] + grad_k1_from_rotate
    grad_k2_total = grad_from_cos_term[..., half_dim:] + grad_k2_from_rotate

    grad_key_states = torch.cat([grad_k1_total, grad_k2_total], dim=-1)

    grad_cos_expanded = grad_key_states_rotated * key_states
    grad_cos = grad_cos_expanded.sum(dim=1)

    grad_sin_expanded = grad_key_states_rotated * k_rotated_half
    grad_sin = grad_sin_expanded.sum(dim=1)

    # Functional construction of cache inputs (avoid in-place index scatter):
    # grad_key_cache_input = grad_key_cache with positions cache_position zeroed.
    # Since cache_position = arange(new_seq_len), zero out [:, :, :new_seq_len].
    nsl = key_states.shape[2]
    gkc_tail = grad_key_cache[:, :, nsl:]
    gvc_tail = grad_value_cache[:, :, nsl:]
    zero_head = torch.zeros_like(grad_key_cache[:, :, :nsl])
    grad_key_cache_input = torch.cat([zero_head, gkc_tail], dim=2)
    zero_head_v = torch.zeros_like(grad_value_cache[:, :, :nsl])
    grad_value_cache_input = torch.cat([zero_head_v, gvc_tail], dim=2)

    return (
        grad_key_states.to(torch.bfloat16),
        grad_value_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_key_cache_input.to(torch.bfloat16),
        grad_value_cache_input.to(torch.bfloat16),
    )

_run_compiled = torch.compile(_run_impl, mode="max-autotune", fullgraph=True)

@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    return _run_compiled(grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position)
