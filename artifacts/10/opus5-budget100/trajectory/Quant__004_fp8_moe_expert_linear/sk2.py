import sys, torch, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=60, w=25):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


# reduce SK partial buffers (SK, M, 2I) fp32 -> silu -> quant (M, I)
@triton.jit
def red(P, Q, S, M, spk, spm, sqm, ssm, I: tl.constexpr, SK: tl.constexpr,
        BM: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = tl.arange(0, 128)
    mm = rm[:, None] < M
    gp = P + rm[:, None] * spm + (2 * pn * 128 + rn)[None, :]
    up = P + rm[:, None] * spm + ((2 * pn + 1) * 128 + rn)[None, :]
    ag = tl.zeros((BM, 128), dtype=tl.float32)
    au = tl.zeros((BM, 128), dtype=tl.float32)
    for k in tl.static_range(SK):
        ag += tl.load(gp + k * spk, mask=mm, other=0.0)
        au += tl.load(up + k * spk, mask=mm, other=0.0)
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(y), axis=1) * RMAX, 1e-12)
    v = tl.minimum(tl.maximum(y / sc[:, None], -EMAX), EMAX)
    tl.store(Q + rm[:, None] * sqm + (pn * 128 + rn)[None, :], v.to(FP8), mask=mm)
    tl.store(S + pn * ssm + rm, sc, mask=rm < M)


H, I = 3584, 2048
for M in [384, 640, 896, 1024]:
    for sk in [2, 4]:
        P = torch.randn(sk, M, 2 * I, device=dev)
        gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
        gs = torch.empty((I // 128, M), device=dev)
        r = []
        for bm, nw in itertools.product([32, 64, 128], [4, 8]):
            t = bench(lambda bm=bm, nw=nw: red[(triton.cdiv(M, bm), I // 128)](
                P, gq, gs, M, P.stride(0), P.stride(1), gq.stride(0),
                gs.stride(0), I=I, SK=sk, BM=bm, num_warps=nw))
            r.append((t, (bm, nw)))
        r.sort()
        print(f"M={M:5d} SK={sk}: reduce {r[0][0]*1e3:5.1f}us {r[0][1]}",
              flush=True)
