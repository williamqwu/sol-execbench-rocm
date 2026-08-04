import torch, triton, triton.language as tl, time

SHAPES = [(1, 512), (4, 512), (8, 512), (16, 512), (1, 4096), (64, 541), (16, 373), (8, 853)]


# A: current -- D-wide compute, 4 half-width stores
@triton.jit
def kA(fp, cp, sp, n_rows, scale, D: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = r[:, None] < n_rows
    x = tl.load(fp + r[:, None] * D + c[None, :], mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cp + o, co, mask=m)
    tl.store(cp + o + D, co, mask=m)
    tl.store(sp + o, si, mask=m)
    tl.store(sp + o + D, si, mask=m)


# B: 2D-wide, duplicate load via modulo, 2 full-width stores
@triton.jit
def kB(fp, cp, sp, n_rows, scale, D: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c2 = tl.arange(0, 2 * D)
    col = c2 % D
    m = r[:, None] < n_rows
    x = tl.load(fp + r[:, None] * D + col[None, :], mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c2[None, :]
    tl.store(cp + o, co, mask=m)
    tl.store(sp + o, si, mask=m)


# C: flat 1-D indexing over output elements
@triton.jit
def kC(fp, cp, sp, n_out, scale, D: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < n_out
    row = i // (2 * D)
    col = i % D
    x = tl.load(fp + row * D + col, mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    tl.store(cp + i, co, mask=m)
    tl.store(sp + i, si, mask=m)


def gpu_time(fn, reps=20, iters=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (iters * reps) * 1000  # us


def ref(f, sc):
    emb = torch.cat((f, f), dim=-1)
    return (emb.cos() * sc).to(torch.bfloat16), (emb.sin() * sc).to(torch.bfloat16)


D = 64
print(f"{'shape':>12} {'MB':>6} | " + " | ".join(f"{n:>22}" for n in ["A(cur)", "B(2D-wide)", "C(flat)"]))
for (b, s) in SHAPES:
    f = torch.randn(b, s, D, device='cuda')
    n = b * s
    cos = torch.empty(b, s, 2 * D, dtype=torch.bfloat16, device='cuda')
    sin = torch.empty(b, s, 2 * D, dtype=torch.bfloat16, device='cuda')
    rc, rs = ref(f, 1.0)
    byts = n * D * 4 + 2 * n * 2 * D * 2
    line = f"{str((b,s)):>12} {byts/1e6:>6.2f} | "
    outs = []
    for tag, kern in [('A', kA), ('B', kB), ('C', kC)]:
        best = (1e9, None)
        for BR in ([1, 2, 4, 8, 16, 32] if tag != 'C' else [256, 512, 1024, 2048, 4096]):
            for w in [1, 2, 4, 8]:
                cos.zero_(); sin.zero_()
                try:
                    if tag == 'C':
                        nout = n * 2 * D
                        gr = (triton.cdiv(nout, BR),)
                        fn = lambda: kern[gr](f, cos, sin, nout, 1.0, D=D, BLK=BR, num_warps=w, num_stages=1)
                    else:
                        gr = (triton.cdiv(n, BR),)
                        fn = lambda: kern[gr](f, cos, sin, n, 1.0, D=D, BR=BR, num_warps=w, num_stages=1)
                    fn(); torch.cuda.synchronize()
                except Exception:
                    continue
                if not (torch.equal(cos, rc) and torch.equal(sin, rs)):
                    continue
                t = gpu_time(fn)
                if t < best[0]:
                    best = (t, (BR, w))
        outs.append(f"{best[0]:>7.2f}us {str(best[1]):>13}")
    print(line + " | ".join(outs))
