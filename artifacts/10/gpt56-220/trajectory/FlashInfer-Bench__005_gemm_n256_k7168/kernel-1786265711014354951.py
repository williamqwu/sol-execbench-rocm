import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, 7168, BLOCK_K):
        a = tl.load(a_ptr + rows[:, None] * 7168 + k0 + ks[None, :],
                    mask=rows[:, None] < M, other=0.0)
        b = tl.load(b_ptr + cols[:, None] * 7168 + k0 + ks[None, :])
        acc += tl.dot(a, tl.trans(b))
    tl.store(c_ptr + rows[:, None] * 256 + cols[None, :], acc,
             mask=rows[:, None] < M)


def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, 256), device=A.device, dtype=A.dtype)
    _gemm_kernel[(triton.cdiv(M, 16), 4)](
        A, B, C, M=M, BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
    )
    return C
