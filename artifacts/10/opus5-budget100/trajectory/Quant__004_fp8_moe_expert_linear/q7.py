import sys, torch, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv


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


@triton.jit
def rd(S, O, N, BL: tl.constexpr):
    p = tl.program_id(0) * BL + tl.arange(0, BL)
    v = tl.load(S + p)
    tl.store(O + tl.program_id(0), tl.sum(v.to(tl.float32)))


H, I = 3584, 2048
N = 2 * I * H
src = torch.randn(N, device=dev, dtype=torch.bfloat16)
o = torch.empty(N // 1024, device=dev)
r = []
for bl, nw in itertools.product([1024, 2048, 4096, 8192, 16384], [1, 2, 4, 8]):
    t = bench(lambda bl=bl, nw=nw: rd[(N // bl,)](src, o, N, BL=bl, num_warps=nw))
    r.append((t, (bl, nw)))
r.sort()
print("pure bf16 read (no write):")
for t, c in r[:5]:
    print(f"   BL={c[0]:6d} nw={c[1]} {t*1e3:6.1f}us  {N*2/(t/1e3)/1e12:.2f} TB/s")
