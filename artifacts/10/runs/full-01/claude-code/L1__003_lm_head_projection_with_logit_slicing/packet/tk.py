"""Tunable Triton GEMM for [M,2048] @ [2048,102400] bf16, TN layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C,
          M, N, K,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          stride_cm, stride_cn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          GROUP_M: tl.constexpr, EVEN_M: tl.constexpr):
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
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    if EVEN_M:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=offs_cm[:, None] < M)


def gemm(a, b, cfg, c=None):
    """a: [M,K] row-major.  b: [K,N] col-major (i.e. weight.t()).  -> [M,N]"""
    M, K = a.shape
    K2, N = b.shape
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK, GM = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["GROUP_M"]
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kw = {}
    for k in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
        if k in cfg:
            kw[k] = cfg[k]
    _gemm[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        EVEN_M=(M % BM == 0),
        num_warps=cfg.get("num_warps", 8),
        num_stages=cfg.get("num_stages", 2),
        **kw,
    )
    return c
