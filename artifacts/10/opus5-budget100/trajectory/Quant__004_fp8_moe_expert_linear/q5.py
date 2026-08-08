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


# DV 1 = true divide (required for bit-match), 0 = reciprocal multiply
@triton.jit
def wq(W1, Q1, S1, s1n, s1s, IB: tl.constexpr, KB1: tl.constexpr,
       DV: tl.constexpr):
    pid = tl.program_id(0)
    nb1 = pid // KB1
    kb1 = pid % KB1
    half = nb1 // IB
    j = nb1 % IB
    dst = 2 * j + half
    rk1 = kb1 * 128 + tl.arange(0, 128)
    sp = (nb1 * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
    dp = (dst * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
    w = tl.load(W1 + sp).to(tl.float32)
    c = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
    if DV:
        z = w / c
    else:
        z = w * (1.0 / c)
    tl.store(Q1 + dp, tl.minimum(tl.maximum(z, -EMAX), EMAX).to(FP8))
    tl.store(S1 + dst * s1s + kb1, c)


H, I = 3584, 2048
guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16)
o = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
sc = torch.empty((2 * I // 128, H // 128), device=dev)
KB1 = H // 128
B = 2 * I * H * 3
for dv in [1, 0]:
    r = []
    for nw in [1, 2, 4, 8]:
        t = bench(lambda dv=dv, nw=nw: wq[((2 * I // 128) * KB1,)](
            guw, o, sc, guw.stride(0), sc.stride(0), IB=I // 128, KB1=KB1,
            DV=dv, num_warps=nw))
        r.append((t, nw))
    r.sort()
    print(f"DV={dv}: " + " | ".join(
        f"nw{n} {t*1e3:.1f}us {B/(t/1e3)/1e12:.2f}TB/s" for t, n in r[:3]))
