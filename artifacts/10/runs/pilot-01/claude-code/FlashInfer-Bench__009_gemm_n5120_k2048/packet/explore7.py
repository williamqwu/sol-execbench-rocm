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

# split-K into separate slices (no atomics), then reduce
@triton.jit
def sk_part(Ap, Bp, Pp, M, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    KS: tl.constexpr = K // SK
    Aptr = Ap + rm[:, None]*K + (pk*KS + rk)[None, :]
    Bptr = Bp + rn[:, None]*K + (pk*KS + rk)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mmask = rm[:, None] < M
    for _ in tl.range(0, KS // BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK; Bptr += BK
    tl.store(Pp + pk*BM*N + rm[:, None]*N + rn[None, :], acc, mask=mmask)

@triton.jit
def sk_red(Pp, Cp, M, N: tl.constexpr, BM: tl.constexpr, SK: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    m = off // N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for k in tl.static_range(SK):
        acc += tl.load(Pp + k*BM*N + off, mask=off < M*N, other=0.0)
    tl.store(Cp + off, acc.to(tl.float16), mask=off < M*N)

def test_sk(M, BM, BN, BK, SK, nw, ns):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    P = torch.empty(SK, BM, N, device=dev, dtype=torch.float32)
    BLK = 1024
    rg = triton.cdiv(M*N, BLK)
    def f():
        sk_part[(N//BN, SK)](A, Bm, P, M, N, K, BM, BN, BK, SK, num_warps=nw, num_stages=ns)
        sk_red[(rg,)](P, C, M, N, BM, SK, BLK, num_warps=4)
    f(); torch.cuda.synchronize()
    if not torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3): return None
    return gpu_time(f)

print("=== split-K sliced ===")
for M in (1, 16):
    out = []
    for BN, BK, SK, nw, ns in itertools.product([16,32],[64,128,256],[2,4],[1,2,4],[2,3]):
        if (K//SK) % BK: continue
        try:
            t = test_sk(M, 16, BN, BK, SK, nw, ns)
            if t: out.append((t, f"BN{BN} BK{BK} SK{SK} w{nw} s{ns}"))
        except Exception: pass
    out.sort()
    print(f"M={M}:", [(round(t,2), c) for t,c in out[:4]])

# more num_stages on the single-kernel variant
@triton.jit
def gemm_a(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    rm = tl.arange(0, BM); rn = pn*BN + tl.arange(0, BN); rk = tl.arange(0, BK)
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

print("\n=== single kernel, deep stages ===")
for M in (1, 16):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    out = []
    for BN, BK, nw, ns in itertools.product([16,32],[128,256,512],[1,2,4],[2,3,4,5,6]):
        try:
            f = lambda: gemm_a[(N//BN,)](A, Bm, C, M, N, K, 16, BN, BK, num_warps=nw, num_stages=ns)
            f(); torch.cuda.synchronize()
            if not torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3): continue
            out.append((gpu_time(f), f"BN{BN} BK{BK} w{nw} s{ns}"))
        except Exception: pass
    out.sort()
    tb = N*K*2 + M*K*2 + M*N*2
    print(f"M={M}:", [(round(t,2), c) for t,c in out[:6]], f"-> {tb/out[0][0]*1e-3:.0f} GB/s" if out else "")
