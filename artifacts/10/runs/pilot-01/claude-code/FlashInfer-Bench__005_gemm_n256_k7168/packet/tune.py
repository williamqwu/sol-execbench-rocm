"""Local tuning scratchpad (not part of the solution)."""
import itertools
import sys
import time

import torch
import triton
import triton.language as tl

DEV = "cuda:0"
N, K = 256, 7168


# ---------------------------------------------------------------- variant A
# N-parallel, full-K loop, one launch. Few workgroups but zero reduction.
@triton.jit
def k_smallm(A, B, C, M, stride_am, stride_bn, stride_cm,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :]
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :],
             acc.to(tl.float16), mask=m_mask[:, None])


def make_A(BM, BN, BK, ws, ns):
    def f(A, B):
        M = A.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        k_smallm[(N // BN,)](A, B, C, M, A.stride(0), B.stride(0), C.stride(0),
                             BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, K=K,
                             num_warps=ws, num_stages=ns)
        return C
    return f


# ---------------------------------------------------------------- variant B
# split-K into workspace, then reduce kernel (2 launches).
@triton.jit
def k_splitk(A, B, W, M, stride_am, stride_bn, stride_ws,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
             NB: tl.constexpr, K: tl.constexpr, NN: tl.constexpr):
    pid = tl.program_id(0)
    sk = pid // NB
    nb = pid % NB
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    KS: tl.constexpr = K // SPLIT_K
    k0 = sk * KS
    a_ptrs = A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :]
    b_ptrs = B + offs_n[:, None] * stride_bn + (k0 + offs_k)[None, :]
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, KS, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    tl.store(W + sk * stride_ws + offs_m[:, None] * NN + offs_n[None, :], acc,
             mask=m_mask[:, None])


@triton.jit
def k_reduce(W, C, M, stride_ws, stride_cm, SPLIT_K: tl.constexpr,
             BLOCK: tl.constexpr, NN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * NN
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(W + s * stride_ws + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(tl.float16), mask=mask)


def make_B(BM, BN, BK, SK, ws, ns):
    NB = N // BN

    def f(A, B):
        M = A.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        W = torch.empty((SK, BM if M < BM else M, N), device=A.device,
                        dtype=torch.float32)
        k_splitk[(NB * SK,)](A, B, W, M, A.stride(0), B.stride(0), W.stride(0),
                             BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
                             NB=NB, K=K, NN=N, num_warps=ws, num_stages=ns)
        n = M * N
        k_reduce[(triton.cdiv(n, 1024),)](W, C, M, W.stride(0), C.stride(0),
                                          SPLIT_K=SK, BLOCK=1024, NN=N)
        return C
    return f


# ---------------------------------------------------------------- variant C
# classic 2D tiled GEMM for large M
@triton.jit
def k_bigm(A, B, C, M, stride_am, stride_bn, stride_cm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, K: tl.constexpr, NB: tl.constexpr,
           GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BLOCK_M)
    if GROUP_M > 1:
        width = GROUP_M * NB
        gid = pid // width
        first = gid * GROUP_M
        gsize = min(nm - first, GROUP_M)
        pm = first + ((pid % width) % gsize)
        pn = (pid % width) // gsize
    else:
        pm = pid // NB
        pn = pid % NB
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    am = offs_m < M
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=am[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :],
             acc.to(tl.float16), mask=am[:, None])


def make_C(BM, BN, BK, ws, ns, gm=1):
    NB = N // BN

    def f(A, B):
        M = A.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        grid = (triton.cdiv(M, BM) * NB,)
        k_bigm[grid](A, B, C, M, A.stride(0), B.stride(0), C.stride(0),
                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, K=K, NB=NB,
                     GROUP_M=gm, num_warps=ws, num_stages=ns)
        return C
    return f


# ---------------------------------------------------------------- harness
def graph_time(fn, args, iters=50, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn(*args)
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters * 1e3)
    return best


def wall_time(fn, args, iters=50, reps=5):
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


import json, os
_TOL = {}
for _l in open(os.path.join(os.path.dirname(__file__), "workload.jsonl")):
    _w = json.loads(_l)
    _TOL[_w["axes"]["M"]] = _w["tolerance"]


def err(out, exp):
    d = (out.float() - exp.float()).abs()
    return d.max().item()


def passes(out, exp, M):
    """Real harness criterion: atol/rtol combined, 99% of elements must match."""
    t = _TOL[M]
    o, e = out.float(), exp.float()
    d = (o - e).abs()
    ok = d <= (t["max_atol"] + t["max_rtol"] * e.abs())
    r = ok.float().mean().item()
    return r >= t["required_matched_ratio"], r, d.max().item()


def mk(M, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = torch.randn(M, K, generator=g, device=DEV, dtype=torch.float16)
    B = torch.randn(N, K, generator=g, device=DEV, dtype=torch.float16)
    return A, B
