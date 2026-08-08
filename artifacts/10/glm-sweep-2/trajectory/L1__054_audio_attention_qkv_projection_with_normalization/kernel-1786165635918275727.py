import torch

@torch.compile(mode="max-autotune", dynamic=True)
def _fused_rms_norm(query, key, q_norm_weight, k_norm_weight, eps):
    # query, key: [batch, seq, num_heads, head_dim]
    q_variance = query.pow(2).mean(dim=-1, keepdim=True)
    q_rms = q_norm_weight * torch.rsqrt(q_variance + eps)
    query_states = query * q_rms

    k_variance = key.pow(2).mean(dim=-1, keepdim=True)
    k_rms = k_norm_weight * torch.rsqrt(k_variance + eps)
    key_states = key * k_rms

    return query_states, key_states


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 8
    head_dim = 128

    query = torch.matmul(hidden_states, q_proj_weight.t())
    key = torch.matmul(hidden_states, k_proj_weight.t())
    value = torch.matmul(hidden_states, v_proj_weight.t())

    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)

    query_states, key_states = _fused_rms_norm(query, key, q_norm_weight, k_norm_weight, eps)

    return query_states, key_states, value
