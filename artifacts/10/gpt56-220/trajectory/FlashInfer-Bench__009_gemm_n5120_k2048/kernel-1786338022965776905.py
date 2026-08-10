import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pn = pid % tl.cdiv(N, BN)
    pm = pid // tl.cdiv(N, BN)
    im = pm * BM + tl.arange(0, BM)
    jn = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + im[:, None] * K + (k + kk[None, :]), mask=im[:, None] < M, other=0.0)
        b = tl.load(B + jn[None, :] * K + (k + kk[:, None]), mask=jn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + im[:, None] * N + jn[None, :], acc, mask=(im[:, None] < M) & (jn[None, :] < N))


def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, 5120), device=A.device, dtype=A.dtype)
    BM = 16 if M <= 64 else 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(5120, 64),)
    _gemm[grid](A, B, C, M=M, N=5120, K=2048, BM=BM, BN=64, BK=32,
                num_warps=4, waves_per_eu=2)
    return C
