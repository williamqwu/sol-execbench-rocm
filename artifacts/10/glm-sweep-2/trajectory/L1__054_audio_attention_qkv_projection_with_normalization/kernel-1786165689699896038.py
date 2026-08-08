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

    # Concatenate QKV weights along output dim -> single GEMM
    # weights are each [qkv_out=1024, hidden=1024]
    # stacked: [3072, 1024]; we compute hidden @ stacked.T -> [batch, seq, 3072]
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = torch.matmul(hidden_states, qkv_weight.t())
    query, key, value = qkv.split(1024, dim=-1)

    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)

    # Per-head RMS normalization for query
    q_variance = query.pow(2).mean(dim=-1, keepdim=True)
    query_states = query * (q_norm_weight * torch.rsqrt(q_variance + eps))

    # Per-head RMS normalization for key
    k_variance = key.pow(2).mean(dim=-1, keepdim=True)
    key_states = key * (k_norm_weight * torch.rsqrt(k_variance + eps))

    return query_states, key_states, value
