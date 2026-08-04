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

print("=== AMD flags, small M ===")
for M in (1, 16):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    out = []
    for BN, BK, nw, ns, wpe, nkd in itertools.product(
            [16, 32], [128, 256], [1, 2, 4], [2, 3, 4], [1, 2, 4, 8], [16]):
        try:
            kw = dict(num_warps=nw, num_stages=ns, waves_per_eu=wpe,
                      matrix_instr_nonkdim=nkd)
            f = lambda: gemm_a[(N//BN,)](A, Bm, C, M, N, K, 16, BN, BK, **kw)
            f(); torch.cuda.synchronize()
            if not torch.allclose(C, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3): continue
            out.append((gpu_time(f), f"BN{BN} BK{BK} w{nw} s{ns} wpe{wpe} nkd{nkd}"))
        except Exception: pass
    out.sort()
    print(f"M={M}:", [(round(t,2), c) for t,c in out[:6]])

print("\n=== large M: torch vs alternatives ===")
for M in (952, 8828, 12251, 16294):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    t_mm = gpu_time(lambda: torch.matmul(A, Bm.T), iters=20)
    Cout = torch.empty(M, N, device=dev, dtype=torch.float16)
    t_out = gpu_time(lambda: torch.mm(A, Bm.T, out=Cout), iters=20)
    print(f"M={M}: matmul {t_mm:.1f}us ({2*M*N*K/t_mm*1e-6:.0f} TF)  mm_out {t_out:.1f}us")
