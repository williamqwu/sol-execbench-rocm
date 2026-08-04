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
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    ts = []
    for _ in range(5):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)


@triton.jit
def skinny(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    pid = tl.program_id(0)
    pk = tl.program_id(1)
    num_n = N // BN
    pm = pid // num_n
    pn = pid % num_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    KS = K // SK
    Aptr = Ap + rm[:, None] * K + (pk * KS + rk)[None, :]
    Bptr = Bp + rn[:, None] * K + (pk * KS + rk)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mmask = rm[:, None] < M
    for _ in range(0, KS, BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK
        Bptr += BK
    c = acc.to(tl.float16)
    Cptr = Cp + rm[:, None] * N + rn[None, :]
    if SK == 1:
        tl.store(Cptr, c, mask=mmask)
    else:
        tl.atomic_add(Cp + rm[:, None] * N + rn[None, :], acc, mask=mmask)


def make(M, BM, BN, BK, SK, nw, ns):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * (N // BN), SK)
    def f():
        skinny[grid](A, Bm, C, M, N, K, BM, BN, BK, SK, num_warps=nw, num_stages=ns)
    return f, A, C


Ms = [int(x) for x in sys.argv[1:]] or [1, 16, 64, 128]
for M in Ms:
    BM = max(16, triton.next_power_of_2(M)) if M <= 128 else 128
    best = []
    for BN, BK, nw, ns in itertools.product([32, 64, 128, 256], [64, 128, 256], [4, 8], [1, 2]):
        if BM * BN // (nw * 64) < 1: continue
        try:
            f, A, C = make(M, BM, BN, BK, 1, nw, ns)
            f(); torch.cuda.synchronize()
            ref = torch.matmul(A, Bm.T)
            if not torch.allclose(C, ref, atol=0.03, rtol=1e-3):
                continue
            t = gpu_time(f)
            best.append((t, BN, BK, nw, ns))
        except Exception as e:
            continue
    best.sort()
    tb = N*K*2 + M*K*2 + M*N*2
    print(f"M={M} BM={BM}:", [(round(t,2), f"BN{bn} BK{bk} w{w} s{s}") for t,bn,bk,w,s in best[:4]],
          f" -> {tb/best[0][0]*1e-3:.0f} GB/s" if best else "NONE")
