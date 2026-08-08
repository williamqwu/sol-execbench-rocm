import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_qk_kernel(
    q_out_ptr, k_out_ptr,
    q_in_ptr, k_in_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr,
    n_rows, num_heads, head_dim, eps,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (batch, seq) row, all heads.
    # Input layout: [n_rows, num_heads, head_dim] contiguous.
    pid = tl.program_id(0)
    if pid >= n_rows:
        return

    offs_d = tl.arange(0, BLOCK_N)
    mask_d = offs_d < head_dim

    qw = tl.load(q_norm_weight_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    kw = tl.load(k_norm_weight_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    row_base = pid * num_heads * head_dim

    for h in range(num_heads):
        base = row_base + h * head_dim

        qx = tl.load(q_in_ptr + base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        q_mean_sq = tl.sum(qx * qx, axis=0) / head_dim
        q_rstd = 1.0 / tl.sqrt(q_mean_sq + eps)
        qy = qx * q_rstd * qw
        tl.store(q_out_ptr + base + offs_d, qy, mask=mask_d)

        kx = tl.load(k_in_ptr + base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        k_mean_sq = tl.sum(kx * kx, axis=0) / head_dim
        k_rstd = 1.0 / tl.sqrt(k_mean_sq + eps)
        ky = kx * k_rstd * kw
        tl.store(k_out_ptr + base + offs_d, ky, mask=mask_d)


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

    n_rows = batch_size * seq_len
    BLOCK_N = triton.next_power_of_2(head_dim)
    query_states = torch.empty_like(query)
    key_states = torch.empty_like(key)
    _rms_norm_qk_kernel[(n_rows,)](
        query_states, key_states,
        query, key,
        q_norm_weight, k_norm_weight,
        n_rows, num_heads, head_dim, eps,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return query_states, key_states, value
