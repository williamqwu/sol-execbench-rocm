import torch, triton, triton.language as tl, itertools, sys
N, K = 5120, 2048
dev = "cuda:0"
torch.manual_seed(0)
Bm = torch.randn(N, K, device=dev, dtype=torch.float16)

def gpu_time(fn, iters=100):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True); ts=[]
    for _ in range(5):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

# ---- Variant A: MFMA, one block per BN rows of B, full K ----
@triton.jit
def gemm_a(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Aptr = Ap + rm[:, None]*K + rk[None, :]
    Bptr = Bp + rn[:, None]*K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mmask = rm[:, None] < M
    for _ in tl.range(0, K // BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK; Bptr += BK
    tl.store(Cp + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=mmask)

# ---- Variant B: GEMV-style broadcast-multiply-reduce (no MFMA) ----
@triton.jit
def gemv_b(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Bptr = Bp + rn[:, None]*K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in tl.range(0, K // BK):
        b = tl.load(Bptr).to(tl.float32)
        for m in tl.static_range(BM):
            a = tl.load(Ap + m*K + rk, mask=(m < M), other=0.0).to(tl.float32)
            partial = tl.sum(a[None, :] * b, axis=1)
            acc = tl.where((tl.arange(0, BM) == m)[:, None], acc + partial[None, :], acc)
        Bptr += BK
    rm = tl.arange(0, BM)
    tl.store(Cp + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=rm[:, None] < M)

def test_a(M, BM, BN, BK, nw, ns):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    grid = (N // BN,)
    f = lambda: gemm_a[grid](A, Bm, C, M, N, K, BM, BN, BK, num_warps=nw, num_stages=ns)
    f(); torch.cuda.synchronize()
    ok = torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3)
    return (gpu_time(f) if ok else None)

def test_b(M, BM, BN, BK, nw, ns):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    grid = (N // BN,)
    f = lambda: gemv_b[grid](A, Bm, C, M, N, K, BM, BN, BK, num_warps=nw, num_stages=ns)
    f(); torch.cuda.synchronize()
    ok = torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3)
    return (gpu_time(f) if ok else None)

which = sys.argv[1] if len(sys.argv) > 1 else "a"
Ms = [int(x) for x in sys.argv[2:]] or [1, 16]
for M in Ms:
    tb = N*K*2 + M*K*2 + M*N*2
    out = []
    if which == "a":
        for BN, BK, nw, ns in itertools.product([16,32,64,128],[32,64,128,256],[1,2,4,8],[1,2,3]):
            try:
                t = test_a(M, 16, BN, BK, nw, ns)
                if t: out.append((t, f"BN{BN} BK{BK} w{nw} s{ns}"))
            except Exception: pass
    else:
        BM = max(1, triton.next_power_of_2(M))
        for BN, BK, nw, ns in itertools.product([16,32,64],[64,128,256],[2,4,8],[1,2]):
            try:
                t = test_b(M, BM, BN, BK, nw, ns)
                if t: out.append((t, f"BN{BN} BK{BK} w{nw} s{ns}"))
            except Exception: pass
    out.sort()
    print(f"[{which}] M={M}:", [(round(t,2), c) for t,c in out[:5]],
          f"-> {tb/out[0][0]*1e-3:.0f} GB/s" if out else "NONE")
