"""C = A @ B.T   A:[M,7168] f16, B:[256,7168] f16 -> C:[M,256] f16.

B.T is a view, so both operands are read along their contiguous (K) axis: the
whole problem is a row-times-row contraction with no transpose traffic.

The regime changes completely with M.

* Small M (the majority of the workloads: M <= 64). Arithmetic is negligible;
  the cost is reading B (3.67 MB) and filling the machine. A conventional tiled
  GEMM launches ceil(M/BLOCK_M) * (256/BLOCK_N) workgroups -- at M=1 with
  BLOCK_N=16 that is 16 workgroups on a 256-CU GPU, so ~94% of the device is
  idle and the kernel is latency-bound on a single pass over B. Splitting the K
  axis SPLIT_K ways multiplies the workgroup count by SPLIT_K and turns the
  same traffic into a parallel read, then a cheap second pass sums the fp32
  partials. This is worth roughly 2.4x over hipBLASLt in GPU time.

* Large M. There are already plenty of tiles to fill the GPU, split-K only adds
  a round trip through memory, and hipBLASLt's assembly-tuned kernels beat a
  Triton tiled loop by a comfortable margin (measured: 70 us vs 79 us at
  M=11948). The honest thing is to call torch there.

Accumulation is fp32 throughout with a single fp16 rounding at the store, which
matches what torch.matmul does for an fp16 GEMM.
"""

import torch
import triton
import triton.language as tl

N = 256
K = 7168
NN_ = tl.constexpr(256)


@triton.jit
def _splitk_mm(A, B, W, M, sam, sbn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
               NB: tl.constexpr, K: tl.constexpr, NN: tl.constexpr):
    """Partial GEMM over one 1/SPLIT_K slice of K, into fp32 workspace W."""
    pid = tl.program_id(0)
    sk = pid % SPLIT_K
    t = pid // SPLIT_K
    pn = t % NB
    pm = t // NB

    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    KS: tl.constexpr = K // SPLIT_K
    k0 = sk * KS
    am = offs_m < M

    a_ptrs = A + offs_m[:, None] * sam + (k0 + offs_k)[None, :]
    b_ptrs = B + offs_n[:, None] * sbn + (k0 + offs_k)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, KS, BLOCK_K):
        a = tl.load(a_ptrs, mask=am[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    tl.store(W + sk * (M * NN) + offs_m[:, None] * NN + offs_n[None, :], acc,
             mask=am[:, None])


@triton.jit
def _reduce(W, C, n_el, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """Sum the SPLIT_K fp32 partials; one rounding to fp16 at the end."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_el
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(W + s * n_el + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(tl.float16), mask=mask)


# M -> (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps), measured on MI355X.
_SMALL = (16, 16, 64, 16, 2)

# Above this, hipBLASLt wins; see module docstring.
_SPLITK_MAX_M = 64


def run(A, B):
    M = A.shape[0]

    if M > _SPLITK_MAX_M:
        return torch.matmul(A, B.T)

    BM, BN, BK, SK, ws = _SMALL
    NB = N // BN
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    W = torch.empty((SK, M, N), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BM) * NB * SK,)
    _splitk_mm[grid](A, B, W, M, A.stride(0), B.stride(0),
                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
                     NB=NB, K=K, NN=N, num_warps=ws, num_stages=2)

    n_el = M * N
    _reduce[(triton.cdiv(n_el, 1024),)](W, C, n_el, SPLIT_K=SK, BLOCK=1024,
                                         num_warps=4)
    return C
