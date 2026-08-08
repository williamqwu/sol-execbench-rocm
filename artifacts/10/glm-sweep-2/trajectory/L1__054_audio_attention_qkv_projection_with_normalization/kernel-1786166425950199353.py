import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_kernel(
    q_out_ptr, k_out_ptr,
    hidden_ptr, q_weight_ptr, k_weight_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr,
    M, K, N, head_dim, eps,
    stride_hidden_m, stride_hidden_k,
    stride_qw_n, stride_qw_k,
    stride_kw_n, stride_kw_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Computes: q[m, n] = hidden[m, :] @ q_weight[n, :]^T, then RMS norm per head.
    # N = num_heads * head_dim. Each program handles BLOCK_M rows and one head (head_dim outputs).
    pid_m = tl.program_id(0)
    head = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = head * HEAD_DIM + offs_d  # output column indices for this head

    mask_m = offs_m < M

    # Accumulators for Q and K
    q_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    k_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Iterate over K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load hidden: [BLOCK_M, BLOCK_K]
        h = tl.load(hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k,
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load q_weight: [HEAD_DIM, BLOCK_K]  (weight is [N, K], we want rows offs_n)
        qw = tl.load(q_weight_ptr + offs_n[:, None] * stride_qw_n + offs_k[None, :] * stride_qw_k,
                      mask=mask_k[None, :], other=0.0)
        q_acc += tl.dot(h, tl.trans(qw))

        # Load k_weight: [HEAD_DIM, BLOCK_K]
        kw = tl.load(k_weight_ptr + offs_n[:, None] * stride_kw_n + offs_k[None, :] * stride_kw_k,
                      mask=mask_k[None, :], other=0.0)
        k_acc += tl.dot(h, tl.trans(kw))

    # Apply RMS norm: variance over HEAD_DIM, per (m, head)
    # q_acc: [BLOCK_M, HEAD_DIM]
    q_mean_sq = tl.sum(q_acc * q_acc, axis=1) / HEAD_DIM  # [BLOCK_M]
    q_rstd = 1.0 / tl.sqrt(q_mean_sq + eps)  # [BLOCK_M]
    qw_vec = tl.load(q_norm_weight_ptr + offs_d, mask=offs_d < HEAD_DIM, other=0.0)  # [HEAD_DIM]
    q_out = q_acc * q_rstd[:, None] * qw_vec[None, :]

    k_mean_sq = tl.sum(k_acc * k_acc, axis=1) / HEAD_DIM
    k_rstd = 1.0 / tl.sqrt(k_mean_sq + eps)
    kw_vec = tl.load(k_norm_weight_ptr + offs_d, mask=offs_d < HEAD_DIM, other=0.0)
    k_out = k_acc * k_rstd[:, None] * kw_vec[None, :]

    # Store: output is [batch, seq, num_heads, head_dim] = [M, num_heads, head_dim]
    # stride: M*head_dim per head, head_dim per m
    out_offset = offs_m[:, None] * HEAD_DIM + head * HEAD_DIM + offs_d[None, :]
    # Actually: layout [M, num_heads, head_dim], stride_m = num_heads*head_dim, stride_head = head_dim
    # So offset = offs_m * (num_heads*head_dim) + head*head_dim + offs_d
    # But we stored with head*HEAD_DIM above incorrectly for m stride. Fix:
    num_heads = N // HEAD_DIM
    out_offset = offs_m[:, None] * (num_heads * HEAD_DIM) + head * HEAD_DIM + offs_d[None, :]
    mask_out = mask_m[:, None]
    tl.store(q_out_ptr + out_offset, q_out, mask=mask_out)
    tl.store(k_out_ptr + out_offset, k_out, mask=mask_out)


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
    M = batch_size * seq_len
    K = hidden_size  # 1024
    N = num_heads * head_dim  # 1024

    hidden_2d = hidden_states.view(M, K)

    query_states = torch.empty(M, num_heads, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    key_states = torch.empty(M, num_heads, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    BLOCK_M = 16
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), num_heads)
    _fused_qk_norm_kernel[grid](
        query_states, key_states,
        hidden_2d, q_proj_weight, k_proj_weight,
        q_norm_weight, k_norm_weight,
        M, K, N, head_dim, eps,
        hidden_2d.stride(0), hidden_2d.stride(1),
        q_proj_weight.stride(0), q_proj_weight.stride(1),
        k_proj_weight.stride(0), k_proj_weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, HEAD_DIM=head_dim,
        num_warps=4,
    )

    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim)

    value = torch.matmul(hidden_states, v_proj_weight.t())
    value_states = value.view(batch_size, seq_len, num_heads, head_dim)

    return query_states, key_states, value_states
