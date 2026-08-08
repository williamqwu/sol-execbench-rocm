import sys, torch, time, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv


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


# SC 0 none | 1 scalar weight only | 2 per-row act only | 3 both (current)
# 4 = both but act scales hoisted out of loop (static unroll)
@triton.jit
def g1(A, SA, W, SW, Q, M, sam, ssam, swn, sswn, sqm,
       BLOCK_M: tl.constexpr, NUM_K: tl.constexpr, SC: tl.constexpr):
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
        a = tl.load(ap, mask=mm, other=0.0)
        bg = tl.load(gp)
        bu = tl.load(up)
        if SC == 0:
            ag = tl.dot(a, tl.trans(bg), ag)
            au = tl.dot(a, tl.trans(bu), au)
        elif SC == 1:
            sg = tl.load(SW + (2 * pn) * sswn + kb)
            su = tl.load(SW + (2 * pn + 1) * sswn + kb)
            ag = tl.math.fma(tl.dot(a, tl.trans(bg)), sg, ag)
            au = tl.math.fma(tl.dot(a, tl.trans(bu)), su, au)
        elif SC == 2:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            ag = tl.math.fma(tl.dot(a, tl.trans(bg)), sa[:, None], ag)
            au = tl.math.fma(tl.dot(a, tl.trans(bu)), sa[:, None], au)
        else:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            sg = tl.load(SW + (2 * pn) * sswn + kb)
            su = tl.load(SW + (2 * pn + 1) * sswn + kb)
            ag = tl.math.fma(tl.dot(a, tl.trans(bg)), (sa * sg)[:, None], ag)
            au = tl.math.fma(tl.dot(a, tl.trans(bu)), (sa * su)[:, None], au)
        ap += 128
        gp += 128
        up += 128
    rq = pn * 128 + rn
    tl.store(Q + rm[:, None] * sqm + rq[None, :], (ag + au).to(FP8), mask=mm)


H, I = 3584, 2048
for M in [4096]:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(M, H // 128, device=dev) * 0.01
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    FL = 2 * M * 2 * I * H
    print(f"--- M={M} ---")
    for sc in [0, 1, 2, 3]:
        r = []
        for bm, nw, ns, wpe in itertools.product([128, 256], [8], [1, 2, 3], [1, 2]):
            t = bench(lambda: g1[(triton.cdiv(M, bm), I // 128)](
                aq, asc, wq, wsc, gq, M, aq.stride(0), asc.stride(0),
                wq.stride(0), wsc.stride(0), gq.stride(0),
                BLOCK_M=bm, NUM_K=H // 128, SC=sc, num_warps=nw,
                num_stages=ns, waves_per_eu=wpe))
            r.append((t, (bm, nw, ns, wpe)))
        r.sort()
        print(f"  SC={sc}: " +
              " | ".join(f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF" for t, c in r[:3]))
