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
    for _ in range(5):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

# split-K with fp32 atomic accumulation into a fp32 buffer, then cast
@triton.jit
def sk_kernel(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    pid = tl.program_id(0); pk = tl.program_id(1)
    num_n = N // BN
    pm = pid // num_n; pn = pid % num_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    KS: tl.constexpr = K // SK
    Aptr = Ap + rm[:, None]*K + (pk*KS + rk)[None, :]
    Bptr = Bp + rn[:, None]*K + (pk*KS + rk)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mmask = rm[:, None] < M
    for _ in tl.range(0, KS, BK):
        a = tl.load(Aptr, mask=mmask, other=0.0)
        b = tl.load(Bptr)
        acc += tl.dot(a, b.T, out_dtype=tl.float32)
        Aptr += BK; Bptr += BK
    tl.atomic_add(Cp + rm[:, None]*N + rn[None, :], acc, mask=mmask)

def run_sk(M, BM, BN, BK, SK, nw, ns):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    C32 = torch.zeros(M, N, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM)*(N//BN), SK)
    def f():
        C32.zero_()
        sk_kernel[grid](A, Bm, C32, M, N, K, BM, BN, BK, SK, num_warps=nw, num_stages=ns)
        return C32.to(torch.float16)
    return f, A, C32

print("=== split-K (atomic fp32) ===")
for M in (1, 16, 64):
    BM = 16 if M <= 16 else 64
    out = []
    for BN, BK, SK, nw, ns in itertools.product([32,64,128],[64,128,256],[2,4,8],[4,8],[2]):
        if (K//SK) % BK: continue
        try:
            f, A, C = run_sk(M, BM, BN, BK, SK, nw, ns)
            r = f(); torch.cuda.synchronize()
            if not torch.allclose(r, torch.matmul(A, Bm.T), atol=0.03, rtol=1e-3): continue
            out.append((gpu_time(f), BN, BK, SK, nw))
        except Exception: continue
    out.sort()
    tb = N*K*2 + M*K*2 + M*N*2
    print(f"M={M}:", [(round(t,2), f"BN{bn} BK{bk} SK{sk} w{w}") for t,bn,bk,sk,w in out[:4]],
          f"-> {tb/out[0][0]*1e-3:.0f} GB/s" if out else "NONE")
