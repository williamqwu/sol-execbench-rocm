import torch
import triton
import triton.language as tl


@triton.jit
def _blockscale_gemm(
    A, B, SA, SB, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sam, stride_sak,
    stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    sa_ptrs = SA + offs_m * stride_sam
    sb_ptr = SB + (pid_n * BLOCK_N // 128) * stride_sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        p = tl.dot(a, b, out_dtype=tl.float32)
        sa = tl.load(sa_ptrs + kb * stride_sak)
        sb = tl.load(sb_ptr + kb * stride_sbk)
        acc += p * (sa[:, None] * sb)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


@torch.no_grad()
def run(hidden_states, weight, scale_x, scale_w):
    batch_size, seq_len, K = hidden_states.shape
    N = weight.shape[0]
    M = batch_size * seq_len

    a = hidden_states.reshape(M, K)
    sa = scale_x.reshape(M, -1)
    # scale_w: (K//128, N//128) -> want (N//128, K//128) view without copy
    sb = scale_w

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _blockscale_gemm[grid](
        a, weight, sa, sb, c,
        M, N, K,
        a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(1),
        sa.stride(0), sa.stride(1),
        sb.stride(1), sb.stride(0),   # sb indexed [n_blk, k_blk] on a (k_blk, n_blk) tensor
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
        num_warps=8, num_stages=2,
    )
    return c.view(batch_size, seq_len, N)
