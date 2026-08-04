import torch, triton, triton.language as tl, itertools
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

# GEMV: acc[m,n] = sum_k A[m,k]*B[n,k], via explicit broadcast, BM tiny
@triton.jit
def gemv(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Bptr = Bp + rn[:, None]*K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    koff = 0
    for _ in tl.range(0, K // BK):
        b = tl.load(Bptr).to(tl.float32)           # (BN, BK)
        for m in tl.static_range(BM):
            a = tl.load(Ap + m*K + koff + rk, mask=(m < M), other=0.0).to(tl.float32)
            p = tl.sum(a[None, :] * b, axis=1)     # (BN,)
            acc += tl.where((tl.arange(0, BM) == m)[:, None], p[None, :], 0.0)
        Bptr += BK
        koff += BK
    rm = tl.arange(0, BM)
    tl.store(Cp + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=rm[:, None] < M)

print("=== GEMV (VALU) ===")
for M in (1, 2, 4, 8):
    BM = triton.next_power_of_2(M)
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    out = []
    for BN, BK, nw, ns in itertools.product([8,16,32,64],[64,128,256],[1,2,4,8],[1,2,3]):
        try:
            f = lambda: gemv[(N//BN,)](A, Bm, C, M, N, K, BM, BN, BK, num_warps=nw, num_stages=ns)
            f(); torch.cuda.synchronize()
            if not torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3): continue
            out.append((gpu_time(f), f"BN{BN} BK{BK} w{nw} s{ns}"))
        except Exception: pass
    out.sort()
    tb = N*K*2 + M*K*2 + M*N*2
    print(f"M={M} BM={BM}:", [(round(t,2), c) for t,c in out[:5]],
          f"-> {tb/out[0][0]*1e-3:.0f} GB/s" if out else "NONE")
