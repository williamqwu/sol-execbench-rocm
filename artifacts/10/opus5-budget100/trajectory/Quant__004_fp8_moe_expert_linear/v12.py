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


# UN=2: two k-blocks per iteration into parity-split accumulators, so the
# fma chain on acc0 is independent of the one on acc1.
@triton.jit
def g1u(A, SA, W, SW, Q, S, M, sam, ssam, swn, sswn, sqm, ssm,
        BLOCK_M: tl.constexpr, NUM_K: tl.constexpr, UN: tl.constexpr):
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
    ag = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    bg2 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    bu2 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K, UN):
        a0 = tl.load(ap, mask=mm, other=0.0)
        g0 = tl.load(gp)
        u0 = tl.load(up)
        sa0 = tl.load(SA + rm * ssam + kb, mask=rv, other=0.0)
        sg0 = tl.load(SW + (2 * pn) * sswn + kb)
        su0 = tl.load(SW + (2 * pn + 1) * sswn + kb)
        ag = tl.math.fma(tl.dot(a0, tl.trans(g0)), (sa0 * sg0)[:, None], ag)
        au = tl.math.fma(tl.dot(a0, tl.trans(u0)), (sa0 * su0)[:, None], au)
        if UN == 2:
            a1 = tl.load(ap + 128, mask=mm, other=0.0)
            g1 = tl.load(gp + 128)
            u1 = tl.load(up + 128)
            sa1 = tl.load(SA + rm * ssam + kb + 1, mask=rv, other=0.0)
            sg1 = tl.load(SW + (2 * pn) * sswn + kb + 1)
            su1 = tl.load(SW + (2 * pn + 1) * sswn + kb + 1)
            bg2 = tl.math.fma(tl.dot(a1, tl.trans(g1)), (sa1 * sg1)[:, None], bg2)
            bu2 = tl.math.fma(tl.dot(a1, tl.trans(u1)), (sa1 * su1)[:, None], bu2)
        ap += 128 * UN
        gp += 128 * UN
        up += 128 * UN
    ag += bg2
    au += bu2
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
    asc = torch.rand(M, H // 128, device=dev) * 0.01
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), device=dev)
    FL = 2 * M * 2 * I * H
    r = []
    for un, bm, nw, ns, wpe in itertools.product(
            [2], [32, 64, 128], [4, 8], [1, 2, 3], [1, 2, 4]):
        if bm * 256 // (nw * 64) > 256 or bm * 256 < nw * 64 * 4:
            continue
        try:
            k = g1u[(triton.cdiv(M, bm), I // 128)](
                aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                BLOCK_M=bm, NUM_K=H // 128, UN=un, num_warps=nw,
                num_stages=ns, waves_per_eu=wpe)
        except Exception:
            continue
        t = bench(lambda: g1u[(triton.cdiv(M, bm), I // 128)](
            aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
            wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
            BLOCK_M=bm, NUM_K=H // 128, UN=un, num_warps=nw, num_stages=ns,
            waves_per_eu=wpe))
        r.append((t, (un, bm, nw, ns, wpe), k.n_regs, k.n_spills))
    r.sort()
    print(f"M={M}: " + "\n       ".join(
        f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF regs={g} sp={s}"
        for t, c, g, s in r[:5]), flush=True)
