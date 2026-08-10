import torch
import triton
import triton.language as tl

@triton.jit
def _gemm(A, B, C, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, 2048, BK):
        a = tl.load(A + offs_m[:, None] * 2048 + k + offs_k[None, :],
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(B + offs_n[None, :] * 2048 + k + offs_k[:, None])
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * 128 + offs_n[None, :], acc,
             mask=offs_m[:, None] < M)

def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, 128), device=A.device, dtype=A.dtype)
    _gemm[(triton.cdiv(M, 32), 2)](A, B, C, M=M, BM=32, BN=64, BK=32,
                                      num_warps=4, num_stages=2)
    return C
