import sys, torch, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=40, w=15):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return 1e9
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


# TR=1 : activation scales stored K-major (KB, M) so the per-kblock read of
#        BLOCK_M scales is contiguous instead of a stride-KB gather.
@triton.jit
def g1t(A, SA, W, SW, Q, S, M, sam, ssam, swn, sswn, sqm, ssm,
        BLOCK_M: tl.constexpr, NUM_K: tl.constexpr, TR: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    gp = W + (2 * pn * 128 + rn)[:, None] * swn + rk[None, :]
    up = W + ((2 * pn + 1) * 128 + rn)[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    rv = rm < M
    if TR:
        sap = SA + rm
    else:
        sap = SA + rm * ssam
    ag = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(ap, mask=mm, other=0.0)
        bg = tl.load(gp)
        bu = tl.load(up)
        sa = tl.load(sap, mask=rv, other=0.0)
        sg = tl.load(SW + (2 * pn) * sswn + kb)
        su = tl.load(SW + (2 * pn + 1) * sswn + kb)
        ag = tl.math.fma(tl.dot(a, tl.trans(bg)), (sa * sg)[:, None], ag)
        au = tl.math.fma(tl.dot(a, tl.trans(bu)), (sa * su)[:, None], au)
        ap += 128
        gp += 128
        up += 128
        if TR:
            sap += ssam
        else:
            sap += 1
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(y), axis=1) * RMAX, 1e-12)
    v = tl.minimum(tl.maximum(y / sc[:, None], -EMAX), EMAX)
    rq = pn * 128 + rn
    tl.store(Q + rm[:, None] * sqm + rq[None, :], v.to(FP8), mask=mm)
    tl.store(S + rm * ssm + pn, sc, mask=rv)


H, I = 3584, 2048
for M in [1024, 2048, 4096]:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    ascN = torch.rand(M, H // 128, device=dev) * 0.01     # M-major (current)
    ascT = torch.rand(H // 128, M, device=dev) * 0.01     # K-major
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), device=dev)
    FL = 2 * M * 2 * I * H
    print(f"--- M={M} ---", flush=True)
    for tr in [0, 1]:
        sa_t = ascT if tr else ascN
        ss = M if tr else H // 128
        r = []
        for bm, nw, ns, wpe in itertools.product([64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
            if bm * 256 // (nw * 64) > 256:
                continue
            t = bench(lambda: g1t[(triton.cdiv(M, bm), I // 128)](
                aq, sa_t, wq, wsc, gq, gs, M, aq.stride(0), ss,
                wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                BLOCK_M=bm, NUM_K=H // 128, TR=tr, num_warps=nw,
                num_stages=ns, waves_per_eu=wpe))
            r.append((t, (bm, nw, ns, wpe)))
        r.sort()
        print(f"  TR={tr}: " + " | ".join(
            f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF" for t, c in r[:3]), flush=True)
