import sys, torch, time, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=40, w=15):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        return 1e9
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


# MASK 1 = current masked loads ; 0 = padded, no mask in loop
@triton.jit
def g1(A, SA, W, SW, Q, S, M, sam, ssam, swn, sswn, sqm, ssm,
       BLOCK_M: tl.constexpr, NUM_K: tl.constexpr, MASK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    gp = W + (2 * pn * 128 + rn)[:, None] * swn + rk[None, :]
    up = W + ((2 * pn + 1) * 128 + rn)[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    ag = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        if MASK:
            a = tl.load(ap, mask=mm, other=0.0)
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
        else:
            a = tl.load(ap)
            sa = tl.load(SA + rm * ssam + kb)
        bg = tl.load(gp)
        bu = tl.load(up)
        sg = tl.load(SW + (2 * pn) * sswn + kb)
        su = tl.load(SW + (2 * pn + 1) * sswn + kb)
        ag = tl.math.fma(tl.dot(a, tl.trans(bg)), (sa * sg)[:, None], ag)
        au = tl.math.fma(tl.dot(a, tl.trans(bu)), (sa * su)[:, None], au)
        ap += 128
        gp += 128
        up += 128
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(y), axis=1) * RMAX, 1e-12)
    v = tl.minimum(tl.maximum(y / sc[:, None], -EMAX), EMAX)
    rq = pn * 128 + rn
    if MASK:
        tl.store(Q + rm[:, None] * sqm + rq[None, :], v.to(FP8), mask=mm)
        tl.store(S + rm * ssm + pn, sc, mask=rm < M)
    else:
        tl.store(Q + rm[:, None] * sqm + rq[None, :], v.to(FP8))
        tl.store(S + rm * ssm + pn, sc)


H, I = 3584, 2048
for M in [1024, 2048, 4096]:
    MP = 4096 + 256
    aq = torch.randint(0, 200, (MP, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(MP, H // 128, device=dev) * 0.01
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((MP, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((MP, I // 128), device=dev)
    FL = 2 * M * 2 * I * H
    print(f"--- M={M} ---")
    for mk in [1, 0]:
        r = []
        for bm, nw, ns, wpe in itertools.product([64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
            if bm * 256 // (nw * 64) > 256:
                continue
            t = bench(lambda: g1[(triton.cdiv(M, bm), I // 128)](
                aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                BLOCK_M=bm, NUM_K=H // 128, MASK=mk, num_warps=nw,
                num_stages=ns, waves_per_eu=wpe))
            r.append((t, (bm, nw, ns, wpe)))
        r.sort()
        print(f"  MASK={mk}: " +
              " | ".join(f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF" for t, c in r[:3]))
