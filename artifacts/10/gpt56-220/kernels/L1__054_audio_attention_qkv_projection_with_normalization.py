import torch
import triton
import triton.language as tl

@triton.jit
def _rms_qk_kernel(q, k, qw, kw, eps: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    qv = tl.load(q + row * N + offs)
    kv = tl.load(k + row * N + offs)
    qi = tl.rsqrt(tl.sum(qv * qv, axis=0) / N + eps)
    ki = tl.rsqrt(tl.sum(kv * kv, axis=0) / N + eps)
    tl.store(q + row * N + offs, qv * qi * tl.load(qw + offs))
    tl.store(k + row * N + offs, kv * ki * tl.load(kw + offs))

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
    """
    Fused QKV projection with per-head RMS normalization.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size=1024]
        q_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        k_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        v_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        q_norm_weight: [head_dim=128]
        k_norm_weight: [head_dim=128]
        eps: epsilon for RMS norm
    
    Returns:
        query_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        key_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        value_states: [batch_size, seq_len, num_heads=8, head_dim=128]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 8
    head_dim = 128
    
    # Linear projections: [batch, seq, hidden] @ [hidden, qkv_out].T -> [batch, seq, qkv_out]
    # Using matmul with transposed weights
    hidden_2d = hidden_states.view(-1, hidden_size)
    query = torch.mm(hidden_2d, q_proj_weight.t())
    key = torch.mm(hidden_2d, k_proj_weight.t())
    value = torch.mm(hidden_2d, v_proj_weight.t())
    
    # Reshape to multi-head format: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)
    
    # Per-head RMS normalization for query
    # Compute variance over head_dim (last dimension)
    rows = batch_size * seq_len * num_heads
    _rms_qk_kernel[(rows,)](
        query, key, q_norm_weight, k_norm_weight, eps=eps, N=head_dim,
        num_warps=1, num_stages=1,
    )
    query_states, key_states = query, key
    
    # No RMS normalization for value in audio attention
    value_states = value
    
    return query_states, key_states, value_states
