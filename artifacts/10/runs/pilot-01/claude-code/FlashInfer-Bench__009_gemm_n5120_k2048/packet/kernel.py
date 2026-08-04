import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_skinny(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[M,N] = A[M,K] @ B[N,K].T  for small M (single tile along M)."""
    pn = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Aptr = Ap + rm[:, None] * K + rk[None, :]
    Bptr = Bp + rn[:, None] * K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mmask = rm[:, None] < M
    for _ in tl.range(0, K // BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK
        Bptr += BK
    tl.store(Cp + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=mmask)


def run(A, B):
    M, K = A.shape
    N = B.shape[0]
    if (M <= 16 and K % 256 == 0 and N % 16 == 0
            and A.is_contiguous() and B.is_contiguous()
            and A.dtype == torch.float16 and B.dtype == torch.float16):
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        _gemm_skinny[(N // 16,)](
            A, B, C, M, N, K,
            BM=16, BN=16, BK=256,
            num_warps=2, num_stages=3, waves_per_eu=1,
        )
        return C
    return torch.matmul(A, B.T)
