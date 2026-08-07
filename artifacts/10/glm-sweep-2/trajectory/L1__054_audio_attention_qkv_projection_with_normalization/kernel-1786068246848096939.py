import torch

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

    # Fuse the three projection GEMMs into one: they share the same LHS (hidden_states).
    # Stacking weights on the output dimension lets the single GEMM read hidden_states once.
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)  # [3*qkv_out, hidden]
    qkv = torch.matmul(hidden_states, qkv_weight.t())  # [batch, seq, 3*qkv_out]

    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    query = qkv[:, :, 0]
    key = qkv[:, :, 1]
    value = qkv[:, :, 2]

    # Per-head RMS normalization for query
    q_variance = query.pow(2).mean(dim=-1, keepdim=True)
    q_rms = torch.rsqrt(q_variance + eps)
    query_states = (query * q_rms) * q_norm_weight

    # Per-head RMS normalization for key
    k_variance = key.pow(2).mean(dim=-1, keepdim=True)
    k_rms = torch.rsqrt(k_variance + eps)
    key_states = (key * k_rms) * k_norm_weight

    # value needs no normalization but must be contiguous in the [batch, seq, num_heads, head_dim] layout.
    value_states = value.contiguous()

    return query_states, key_states, value_states
