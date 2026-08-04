"""General split-K + tiled variants, measured both GPU-only and wall-clock."""
import time
import torch
import triton
import triton.language as tl

DEV = "cuda:0"
N, K = 256, 7168
NN_ = tl.constexpr(256)


# ---- general split-K: grid = (ceil(M/BM)*NB*SK,), workspace fp32 ----
@triton.jit
def gsk(A, B, W, M, sam, sbn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr, NB: tl.constexpr, K: tl.constexpr,
        NN: tl.constexpr):
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
def gred(W, C, n_el, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_el
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(W + s * n_el + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(tl.float16), mask=mask)


def make_gsk(BM, BN, BK, SK, ws, ns):
    NB = N // BN

    def f(A, B):
        M = A.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        W = torch.empty((SK, M, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BM) * NB * SK,)
        gsk[grid](A, B, W, M, A.stride(0), B.stride(0),
                  BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK, NB=NB, K=K,
                  NN=N, num_warps=ws, num_stages=ns)
        n_el = M * N
        gred[(triton.cdiv(n_el, 1024),)](W, C, n_el, SPLIT_K=SK, BLOCK=1024,
                                          num_warps=4)
        return C
    return f


# ---- plain tiled, no split-K, one launch ----
@triton.jit
def tiled(A, B, C, M, sam, sbn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          NB: tl.constexpr, K: tl.constexpr, EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    pn = pid % NB
    pm = pid // NB
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :]
    b_ptrs = B + offs_n[:, None] * sbn + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if EVEN_M:
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, tl.trans(b), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        tl.store(C + offs_m[:, None] * NN_ + offs_n[None, :], acc.to(tl.float16))
    else:
        am = offs_m < M
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=am[:, None], other=0.0)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, tl.trans(b), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        tl.store(C + offs_m[:, None] * NN_ + offs_n[None, :], acc.to(tl.float16),
                 mask=am[:, None])


def make_tiled(BM, BN, BK, ws, ns):
    NB = N // BN

    def f(A, B):
        M = A.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        grid = (triton.cdiv(M, BM) * NB,)
        tiled[grid](A, B, C, M, A.stride(0), B.stride(0),
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, NB=NB, K=K,
                    EVEN_M=(M % BM == 0), num_warps=ws, num_stages=ns)
        return C
    return f


# ---------------- timing ----------------
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


def event_time(fn, args, iters=50, reps=7):
    """Mimics do_bench: enqueue iters back-to-back, event-bracketed."""
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters):
            fn(*args)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


import json, os
_TOL = {}
for _l in open(os.path.join(os.path.dirname(__file__), "workload.jsonl")):
    _w = json.loads(_l)
    _TOL[_w["axes"]["M"]] = _w["tolerance"]


def passes(out, exp, M):
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
