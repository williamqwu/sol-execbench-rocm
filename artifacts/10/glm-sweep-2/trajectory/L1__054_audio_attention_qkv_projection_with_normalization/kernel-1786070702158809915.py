import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_rmsnorm_kernel(
    # pointers
    a_ptr, w_ptr, norm_w_ptr, out_ptr,
    # strides for a: [M, K]
    stride_am, stride_ak,
    # strides for w: [N, K]  (weight already [N,K], we compute a @ w.T)
    stride_wn, stride_wk,
    # strides for out: [M, N]
    stride_om, stride_on,
    # dims
    M, N: tl.constexpr, K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # one tile per head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * HEAD_DIM + tl.arange(0, HEAD_DIM)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator: [BLOCK_M, HEAD_DIM] fp32
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # a block: [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # w block: [HEAD_DIM, BLOCK_K]  (w is [N, K], we want head row)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        w = tl.load(w_ptrs)  # HEAD_DIM x BLOCK_K
        acc += tl.dot(a, w.trans(1, 0))
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # RMSNorm over HEAD_DIM
    mean_sq = tl.sum(acc * acc, axis=1, keep_dims=True) / HEAD_DIM
    rms_inv = 1.0 / tl.sqrt(mean_sq + EPS)
    norm_w = tl.load(norm_w_ptr + tl.arange(0, HEAD_DIM))
    acc = (acc * rms_inv) * norm_w[None, :]

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = offs_m[:, None] < M
    tl.store(out_ptrs, acc, mask=mask)


def _gemm_rmsnorm(a, w, norm_w, eps):
    # a: [M, K], w: [N, K], norm_w: [HEAD_DIM], out: [M, N]
    M, K = a.shape
    N, _ = w.shape
    HEAD_DIM = norm_w.shape[0]
    out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    BLOCK_M = 16 if M < 64 else 64
    grid = (triton.cdiv(M, BLOCK_M), N // HEAD_DIM)
    _gemm_rmsnorm_kernel[grid](
        a, w, norm_w, out,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        M, N, K, HEAD_DIM, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=HEAD_DIM, BLOCK_K=128,
        num_warps=4,
    )
    return out


@torch.no_grad()
def run(
    hidden_states, q_proj_weight, k_proj_weight, v_proj_weight,
    q_norm_weight, k_norm_weight, eps,
):
    bs, sl, _ = hidden_states.shape
    nh, hd = 8, 128
    M = bs * sl
    a = hidden_states.view(M, -1)

    query_states = _gemm_rmsnorm(a, q_proj_weight, q_norm_weight, eps).view(bs, sl, nh, hd)
    key_states = _gemm_rmsnorm(a, k_proj_weight, k_norm_weight, eps).view(bs, sl, nh, hd)
    value_states = torch.matmul(hidden_states, v_proj_weight.t()).view(bs, sl, nh, hd)

    return query_states, key_states, value_states
