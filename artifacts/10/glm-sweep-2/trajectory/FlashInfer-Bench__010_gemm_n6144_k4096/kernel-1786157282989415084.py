import torch
import triton
import triton.language as tl


@triton.jit
def _gemv1_kernel(A, B, C, N, K, stride_bn,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(A + offs_k, mask=k_mask, other=0.0)
        b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :]
        b = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(a[None, :].to(tl.float32) * b.to(tl.float32), axis=1)
    tl.store(C + offs_n, acc.to(tl.float16), mask=n_mask)


def _gemv1(A, B):
    N, K = B.shape
    C = torch.empty(1, N, dtype=torch.float16, device=A.device)
    grid = (triton.cdiv(N, 16),)
    _gemv1_kernel[grid](A, B, C, N, K, B.stride(0),
                        BLOCK_N=16, BLOCK_K=2048, num_warps=2, num_stages=3)
    return C


def run(A, B):
    if A.shape[0] == 1:
        return _gemv1(A, B)
    return torch.matmul(A, B.T)
