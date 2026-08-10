import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_group) % group_size_m)
    pid_n = (pid % num_pid_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        av = tl.load(a + offs_m[:, None] * K + (k + offs_k[None, :]),
                     mask=offs_m[:, None] < M, other=0.0)
        bv = tl.load(b + offs_n[None, :] * K + (k + offs_k[:, None]))
        acc += tl.dot(av, bv)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, B.shape[0]), device=A.device, dtype=A.dtype)
    bm, bn, bk = 32, 64, 32
    grid = (triton.cdiv(M, bm) * triton.cdiv(B.shape[0], bn),)
    _gemm_kernel[grid](A, B, C, M, B.shape[0], A.shape[1], bm, bn, bk, 8,
                       num_warps=4, num_stages=2)
    return C
