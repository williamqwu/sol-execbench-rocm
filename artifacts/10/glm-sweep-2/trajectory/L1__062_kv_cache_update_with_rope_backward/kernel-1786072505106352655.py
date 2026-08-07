import torch

@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    half_dim = key_states.shape[-1] // 2
    nsl = key_states.shape[2]

    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    # cache_position == arange(nsl), so gather == contiguous slice
    grad_key_states_rotated = grad_key_cache[:, :, :nsl]
    grad_value_states = grad_value_cache[:, :, :nsl]

    grad_from_cos_term = grad_key_states_rotated * cos_expanded
    grad_k_rotated_half = grad_key_states_rotated * sin_expanded

    grad_k1_total = grad_from_cos_term[..., :half_dim] + grad_k_rotated_half[..., half_dim:]
    grad_k2_total = grad_from_cos_term[..., half_dim:] - grad_k_rotated_half[..., :half_dim]

    grad_key_states = torch.cat([grad_k1_total, grad_k2_total], dim=-1)

    grad_cos = (grad_key_states_rotated * key_states).sum(dim=1)
    grad_sin = (grad_key_states_rotated * k_rotated_half).sum(dim=1)

    # Avoid clone+scatter: zero the head positions, copy the tail.
    head_k = grad_key_cache[:, :, :nsl]
    tail_k = grad_key_cache[:, :, nsl:]
    grad_key_cache_input = torch.cat([torch.zeros_like(head_k), tail_k], dim=2)

    head_v = grad_value_cache[:, :, :nsl]
    tail_v = grad_value_cache[:, :, nsl:]
    grad_value_cache_input = torch.cat([torch.zeros_like(head_v), tail_v], dim=2)

    return (
        grad_key_states.to(torch.bfloat16),
        grad_value_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_key_cache_input.to(torch.bfloat16),
        grad_value_cache_input.to(torch.bfloat16),
    )
