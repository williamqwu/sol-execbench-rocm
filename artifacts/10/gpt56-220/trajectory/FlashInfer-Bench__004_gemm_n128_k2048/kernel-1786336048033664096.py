import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _small_gemm(A, B, C, M: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    ks = tl.arange(0, BK)
    acc = tl.zeros((16, 128), tl.float32)
    for k in range(0, 2048, BK):
        a = tl.load(A + rows[:, None] * 2048 + k + ks[None, :],
                    mask=rows[:, None] < M, other=0.0)
        b = tl.load(B + cols[None, :] * 2048 + k + ks[:, None])
        acc += tl.dot(a, b)
    tl.store(C + rows[:, None] * 128 + cols[None, :], acc,
             mask=rows[:, None] < M)


def run(A, B):
    M = A.shape[0]
    if M <= 16:
        C = torch.empty((M, 128), dtype=A.dtype, device=A.device)
        _small_gemm[(1,)](A, B, C, M=M, BK=64, num_warps=8, num_stages=2)
        return C
    return F.linear(A, B)
