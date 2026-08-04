import torch, triton, triton.language as tl, time, itertools

D = 64


# A: baseline, 4 half stores
@triton.jit
def kA(fp, cp, sp, n, scale, D: tl.constexpr, BR: tl.constexpr, EV: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = r[:, None] < n
    x = tl.load(fp + r[:, None] * D + c[None, :], mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cp + o, co, mask=m, cache_modifier=EV)
    tl.store(cp + o + D, co, mask=m, cache_modifier=EV)
    tl.store(sp + o, si, mask=m, cache_modifier=EV)
    tl.store(sp + o + D, si, mask=m, cache_modifier=EV)


# D: no masking when grid divides evenly (separate specialization)
@triton.jit
def kD(fp, cp, sp, n, scale, D: tl.constexpr, BR: tl.constexpr, EV: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    x = tl.load(fp + r[:, None] * D + c[None, :])
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cp + o, co, cache_modifier=EV)
    tl.store(cp + o + D, co, cache_modifier=EV)
    tl.store(sp + o, si, cache_modifier=EV)
    tl.store(sp + o + D, si, cache_modifier=EV)


# E: flat contiguous over INPUT elements; each program handles BLK input elems
#    output written as two half-row stores derived from flat index
@triton.jit
def kE(fp, cp, sp, n, scale, D: tl.constexpr, BR: tl.constexpr, EV: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * (BR * D) + tl.arange(0, BR * D)
    m = i < n * D
    x = tl.load(fp + i, mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    row = i // D
    col = i % D
    o = row * (2 * D) + col
    tl.store(cp + o, co, mask=m, cache_modifier=EV)
    tl.store(cp + o + D, co, mask=m, cache_modifier=EV)
    tl.store(sp + o, si, mask=m, cache_modifier=EV)
    tl.store(sp + o + D, si, mask=m, cache_modifier=EV)


def gpu_time(fn, reps=20, iters=60):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (iters * reps) * 1000


SHAPES = [(64, 541), (16, 512), (8, 512), (4, 512), (1, 512)]
for (b, s) in SHAPES:
    f = torch.randn(b, s, D, device='cuda')
    n = b * s
    out = torch.empty((2,) + (b, s) + (2 * D,), dtype=torch.bfloat16, device='cuda')
    cos, sin = out.unbind(0)
    emb = torch.cat((f, f), -1)
    rc = emb.cos().to(torch.bfloat16); rs = emb.sin().to(torch.bfloat16)
    byts = n * D * 4 + 2 * n * 2 * D * 2
    sol = byts / 8e12 * 1e6
    res = []
    for tag, kern in [('A', kA), ('D', kD), ('E', kE)]:
        for BR in [1, 2, 4, 8, 16, 32, 64]:
            for w in [1, 2, 4, 8]:
                for ev in ['', '.cs', '.cg', '.wt']:
                    if tag == 'D' and n % BR: continue
                    grid = (triton.cdiv(n, BR),)
                    try:
                        out.zero_()
                        kern[grid](f, cos, sin, n, 1.0, D=D, BR=BR, EV=ev, num_warps=w, num_stages=1)
                        torch.cuda.synchronize()
                    except Exception:
                        continue
                    if not (torch.equal(cos, rc) and torch.equal(sin, rs)):
                        continue
                    t = gpu_time(lambda: kern[grid](f, cos, sin, n, 1.0, D=D, BR=BR, EV=ev, num_warps=w, num_stages=1))
                    res.append((t, tag, BR, w, ev or 'none'))
    res.sort()
    print(f"\n=== {(b,s)}  n={n}  {byts/1e6:.2f}MB  SOL={sol:.2f}us ===")
    for t, tag, BR, w, ev in res[:6]:
        print(f"   {t:7.2f}us  {100*sol/t:5.1f}%SOL  {tag} BR={BR:<3} warps={w} cache={ev}")
