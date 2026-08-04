import torch, triton, triton.language as tl, itertools, sys
N, K = 5120, 2048
dev = "cuda:0"
torch.manual_seed(0)
Bm = torch.randn(N, K, device=dev, dtype=torch.float16)

def gpu_time(fn, iters=50):
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
    for _ in range(3):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

# General 2D-tiled GEMM, B is (N,K) so we do a @ b.T
@triton.jit
def gemm(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = N // BN
    ng = GM * nn
    gid = pid // ng
    fm = gid * GM
    gsz = min(nm - fm, GM)
    pm = fm + ((pid % ng) % gsz)
    pn = (pid % ng) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm[:, None] < M
    Aptr = Ap + rm[:, None]*K + rk[None, :]
    Bptr = Bp + rn[:, None]*K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in tl.range(0, K // BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK; Bptr += BK
    tl.store(Cp + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=mmask)

Ms = [int(x) for x in sys.argv[1:]]
for M in Ms:
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    ref = torch.matmul(A, Bm.T)
    tr = gpu_time(lambda: torch.matmul(A, Bm.T), iters=(50 if M<2000 else 10))
    out = []
    BMs = [16,32,64,128,256] if M > 16 else [16]
    for BM, BN, BK, nw, ns in itertools.product(BMs,[16,32,64,128,256],[32,64,128,256],[1,2,4,8],[1,2,3,4]):
        if BM > 2*triton.next_power_of_2(M): continue
        try:
            for GM in (1, 4, 8):
                f = lambda: gemm[(triton.cdiv(M,BM)*(N//BN),)](A, Bm, C, M, N, K, BM, BN, BK, GM,
                                                               num_warps=nw, num_stages=ns)
                f(); torch.cuda.synchronize()
                if not torch.allclose(C, ref, atol=0.03, rtol=1e-3): continue
                out.append((gpu_time(f, iters=(50 if M<2000 else 10)), f"BM{BM} BN{BN} BK{BK} w{nw} s{ns} GM{GM}"))
        except Exception: pass
    out.sort()
    print(f"M={M} torch={tr:.2f}:", [(round(t,2), c) for t,c in out[:4]], flush=True)
