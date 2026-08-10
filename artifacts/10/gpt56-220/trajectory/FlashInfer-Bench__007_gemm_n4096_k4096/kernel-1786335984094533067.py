import torch
import triton
import triton.language as tl

@triton.jit
def _gemm(a, b, c, M: tl.constexpr, BM: tl.constexpr = 64,
          BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    pid = tl.program_id(0)
    pm = pid // (4096 // BN)
    pn = pid % (4096 // BN)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, 4096, BK):
        av = tl.load(a + rm[:, None] * 4096 + (k + rk[None, :]),
                     mask=rm[:, None] < M, other=0.0)
        bv = tl.load(b + rn[None, :] * 4096 + (k + rk[:, None]))
        acc += tl.dot(av, bv)
    tl.store(c + rm[:, None] * 4096 + rn[None, :], acc,
             mask=rm[:, None] < M)

def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, 4096), device=A.device, dtype=A.dtype)
    _gemm[(triton.cdiv(M, 64) * 64,)](A, B, C, M=M, num_warps=4)
    return C
