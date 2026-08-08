import sys, torch, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=50, w=20):
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


# pure bf16 -> fp8 cast+copy, no reduction: bandwidth ceiling for this traffic
@triton.jit
def cp(S, Dst, N, BL: tl.constexpr):
    p = tl.program_id(0) * BL + tl.arange(0, BL)
    v = tl.load(S + p, mask=p < N, other=0.0).to(tl.float32)
    tl.store(Dst + p, (v * 0.5).to(FP8), mask=p < N)


# quant with the max reduction, flat 1D view of a 128x128 tile
@triton.jit
def qflat(S, Dst, Sc, NT, BL: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BL + tl.arange(0, BL)
    v = tl.load(S + p).to(tl.float32)
    c = tl.maximum(tl.max(tl.abs(v)) * RMAX, 1e-12)
    tl.store(Dst + p, tl.minimum(tl.maximum(v / c, -EMAX), EMAX).to(FP8))
    tl.store(Sc + pid, c)


H, I = 3584, 2048
N = 2 * I * H
src = torch.randn(N, device=dev, dtype=torch.bfloat16)
dst = torch.empty(N, dtype=torch.float8_e4m3fn, device=dev)
sc = torch.empty(N // 16384, device=dev)
BYTES = N * 2 + N
r = []
for bl, nw in itertools.product([1024, 2048, 4096, 8192], [1, 2, 4, 8]):
    t = bench(lambda: cp[(triton.cdiv(N, bl),)](src, dst, N, BL=bl, num_warps=nw))
    r.append((t, ('copy', bl, nw)))
for nw in [1, 2, 4, 8, 16]:
    t = bench(lambda: qflat[(N // 16384,)](src, dst, sc, 1, BL=16384, num_warps=nw))
    r.append((t, ('qflat16k', nw)))
r.sort()
for t, c in r[:10]:
    print(f"   {str(c):22s} {t*1e3:6.1f}us  {BYTES/(t/1e3)/1e12:.2f} TB/s")
