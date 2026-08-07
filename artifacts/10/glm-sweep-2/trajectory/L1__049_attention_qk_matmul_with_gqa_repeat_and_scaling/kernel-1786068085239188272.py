import torch
import triton
import triton.language as tl


@triton.jit
def _qk_matmul_kernel(
    Q_ptr, K_ptr, OUT_ptr, scale,
    B, Hq, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ob, stride_oh, stride_os, stride_os2,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid = tl.program_id(1)  # linear over M*N

    b = pid_bh // Hq
    h = pid_bh % Hq

    num_pid_m = tl.cdiv(S, BM)
    num_pid_n = tl.cdiv(S, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, BK)

    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb

    q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K_ptr + k_base + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, 256, BK):
        q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < S, other=0.0)
        acc += tl.dot(q, k)
        q_ptrs += BK * stride_qd
        k_ptrs += BK * stride_kd

    acc = acc * scale

    o_base = b * stride_ob + h * stride_oh
    o_ptrs = OUT_ptr + o_base + offs_m[:, None] * stride_os + offs_n[None, :] * stride_os2
    tl.store(o_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=(offs_m[:, None] < S) & (offs_n[None, :] < S))


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    B, Hq, S, D = query.shape
    out = torch.empty((B, Hq, S, S), dtype=query.dtype, device=query.device)
    if S == 0:
        return out
    BM, BN, BK = 64, 64, 64
    num_pid_m = triton.cdiv(S, BM)
    num_pid_n = triton.cdiv(S, BN)
    grid = (B * Hq, num_pid_m * num_pid_n)
    _qk_matmul_kernel[grid](
        query, key, out, scaling,
        B, Hq, S,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(2), key.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BM=BM, BN=BN, BK=BK, GROUP_M=8,
    )
    return out
