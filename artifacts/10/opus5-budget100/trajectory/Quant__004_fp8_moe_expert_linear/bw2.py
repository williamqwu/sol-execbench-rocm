import sys, torch, triton, triton.language as tl, itertools
dev = 'cuda'


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
def rd(S, O, BL: tl.constexpr, NPP: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BL,), dtype=tl.float32)
    for i in tl.range(0, NPP):
        p = (pid * NPP + i) * BL + tl.arange(0, BL)
        acc += tl.load(S + p).to(tl.float32)
    tl.store(O + pid * BL + tl.arange(0, BL), acc)


for MB in [29, 128, 512]:
    N = MB * 1024 * 1024 // 2
    src = torch.randn(N, device=dev, dtype=torch.bfloat16)
    r = []
    for bl, npp, nw in itertools.product([1024, 4096], [1, 4, 16], [4, 8]):
        if N % (bl * npp):
            continue
        o = torch.empty(N // npp, device=dev)
        t = bench(lambda bl=bl, npp=npp, nw=nw: rd[(N // (bl * npp),)](
            src, o, BL=bl, NPP=npp, num_warps=nw))
        r.append((t, (bl, npp, nw)))
    r.sort()
    print(f"{MB}MB: " + " | ".join(
        f"{c} {N*2/(t/1e3)/1e12:.2f}TB/s" for t, c in r[:3]), flush=True)
