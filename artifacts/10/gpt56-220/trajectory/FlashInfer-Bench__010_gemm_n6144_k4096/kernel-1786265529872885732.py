import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(a, b, c, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, k, BK):
        av = tl.load(a + offs_m[:, None] * k + kb + offs_k[None, :],
                     mask=offs_m[:, None] < m, other=0.0)
        bv = tl.load(b + offs_n[None, :] * k + kb + offs_k[:, None],
                     mask=offs_n[None, :] < n, other=0.0)
        acc += tl.dot(av, bv)
    tl.store(c + offs_m[:, None] * n + offs_n[None, :], acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def run(A, B):
    m, k = A.shape
    n = B.shape[0]
    out = torch.empty((m, n), device=A.device, dtype=A.dtype)
    _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        A, B, out, m=m, n=n, k=k, BM=32, BN=64, BK=32,
        num_warps=4, num_stages=2)
    return out
