import sys, torch, time, triton, triton.language as tl
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
dev = 'cuda'
FP8 = tl.float8e4nv


def bench(fn, n=60, w=25):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        print("   ERR", str(e)[:120])
        return 1e9
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# MODE: 0=noscale(chained dot), 1=noscale but separate acc add,
# 2=scale const, 3=scale loaded rowvec only, 4=full (row*col scalar)
@triton.jit
def kern(A, SA, W, SW, Q, M, I, sam, ssam, swn, sswn, sqm,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr,
         MODE: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    gp = W + rn[:, None] * swn + rk[None, :]
    up = W + (rn + I)[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    ag = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(ap, mask=mm, other=0.0)
        bg = tl.load(gp)
        bu = tl.load(up)
        if MODE == 0:
            ag = tl.dot(a, tl.trans(bg), ag)
            au = tl.dot(a, tl.trans(bu), au)
        elif MODE == 1:
            ag += tl.dot(a, tl.trans(bg))
            au += tl.dot(a, tl.trans(bu))
        elif MODE == 2:
            ag += tl.dot(a, tl.trans(bg)) * 1.0001
            au += tl.dot(a, tl.trans(bu)) * 1.0001
        elif MODE == 3:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            ag += tl.dot(a, tl.trans(bg)) * sa[:, None]
            au += tl.dot(a, tl.trans(bu)) * sa[:, None]
        else:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            sg = tl.load(SW + pn * sswn + kb)
            su = tl.load(SW + (pn + I // 128) * sswn + kb)
            ag += tl.dot(a, tl.trans(bg)) * (sa[:, None] * sg)
            au += tl.dot(a, tl.trans(bu)) * (sa[:, None] * su)
        ap += 128
        gp += 128
        up += 128
    tl.store(Q + rm[:, None] * sqm + rn[None, :], (ag + au).to(FP8), mask=mm)


H, I, M = 3584, 2048, 4096
aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
asc = torch.rand(M, H // 128, device=dev) * 0.01
wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
FL = 2 * M * 2 * I * H
names = {0: "chained dot (no scale)", 1: "acc+= dot", 2: "*const",
         3: "*rowvec", 4: "*rowvec*colscalar (real)"}
for mode in range(5):
    for bm, bn, nw, ns in [(128, 128, 8, 2), (256, 128, 8, 2), (128, 128, 8, 3)]:
        t = bench(lambda: kern[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
            aq, asc, wq, wsc, gq, M, I, aq.stride(0), asc.stride(0), wq.stride(0),
            wsc.stride(0), gq.stride(0), BLOCK_M=bm, BLOCK_N=bn, NUM_K=H // 128,
            MODE=mode, num_warps=nw, num_stages=ns))
        print(f"MODE{mode} {names[mode]:26s} bm{bm} bn{bn} ns{ns}: "
              f"{t*1e3:6.1f} us {FL/(t/1e3)/1e12:6.0f} TF/s")
