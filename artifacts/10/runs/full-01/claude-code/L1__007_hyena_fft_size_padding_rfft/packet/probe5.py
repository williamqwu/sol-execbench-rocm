import torch, triton, triton.language as tl, time, json
dev = 'cuda:0'


@triton.jit
def eA(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


@triton.jit
def eB(SRC, OUT, M, INV, BLOCK: tl.constexpr, VEC: tl.constexpr):
    # each program handles BLOCK*VEC complex elements
    pid = tl.program_id(0)
    base = pid*BLOCK*VEC
    i = tl.arange(0, BLOCK)[:, None]*VEC + tl.arange(0, VEC)[None, :]   # (BLOCK,VEC)
    idx = base + i
    m = idx < M
    lo = tl.load(SRC + idx[:, :, None]*2 + tl.arange(0, 2)[None, None, :],
                 mask=m[:, :, None], other=0.)
    re, im = tl.split(lo)
    tl.store(OUT + idx, re*INV, mask=m)
    tl.store(OUT + M + idx, im*INV, mask=m)


@triton.jit
def eC(SRC, OUT, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid*BLOCK
    o2 = 2*base + tl.arange(0, 2*BLOCK)
    v = tl.load(SRC + o2, mask=o2 < 2*M, other=0.)
    re, im = tl.split(v.reshape(BLOCK, 2))
    offs = base + tl.arange(0, BLOCK); m = offs < M
    tl.store(OUT + offs, re*INV, mask=m)
    tl.store(OUT + M + offs, im*INV, mask=m)


def gbench(f, iters=50, warm=15):
    for _ in range(warm): f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters


rows_w = [json.loads(l) for l in open('workload.jsonl')]
CASES = sorted({(w['axes']['batch_size'], w['axes']['seqlen']) for w in rows_w})

print(f"{'B':>3}{'S':>7}{'M(K)':>9} | best-config                      | ms      GB/s")
BEST = {}
for (B, S) in CASES:
    C = 256; N = 2*S; F = S+1; M = B*C*F
    x = torch.randn(B, C, S, device=dev, dtype=torch.float32)
    z = torch.fft.rfft(x, n=N)
    src = torch.view_as_real(z)
    out = torch.empty((2, M), device=dev, dtype=torch.float32)
    inv = 1.0/N
    gr = (z/N).real.contiguous().reshape(-1); gi = (z/N).imag.contiguous().reshape(-1)
    cands = []
    for bs in (256, 512, 1024, 2048, 4096):
        for nw in (1, 2, 4, 8):
            cands.append(("A", eA, bs, nw, None))
            cands.append(("C", eC, bs, nw, None))
    for bs in (128, 256, 512, 1024):
        for nw in (2, 4, 8):
            for vec in (2, 4):
                cands.append(("B", eB, bs, nw, vec))
    results = []
    for (nm, kern, bs, nw, vec) in cands:
        try:
            if nm == "A":
                f = lambda kern=kern, bs=bs, nw=nw: kern[(triton.cdiv(M, bs),)](src, out[0], out[1], M, inv, BLOCK=bs, num_warps=nw)
            elif nm == "C":
                f = lambda kern=kern, bs=bs, nw=nw: kern[(triton.cdiv(M, bs),)](src, out, M, inv, BLOCK=bs, num_warps=nw)
            else:
                f = lambda kern=kern, bs=bs, nw=nw, vec=vec: kern[(triton.cdiv(M, bs*vec),)](src, out, M, inv, BLOCK=bs, num_warps=nw, VEC=vec)
            out.zero_(); f(); torch.cuda.synchronize()
            if not (torch.equal(out[0], gr) and torch.equal(out[1], gi)):
                continue
            t = gbench(f)
            results.append((t, nm, bs, nw, vec))
        except Exception:
            continue
    results.sort()
    t, nm, bs, nw, vec = results[0]
    gbs = (16.0*M/1e9)/(t/1e3)
    BEST[(B, S)] = (nm, bs, nw, vec, t)
    print(f"{B:>3}{S:>7}{M/1000:9.1f} | {nm} BLOCK={bs:<5} warps={nw} vec={vec}      | {t:.4f}  {gbs:7.1f}")
    top = results[:4]
    print("      runners-up: " + ", ".join(f"{r[1]}/b{r[2]}/w{r[3]}/v{r[4]}={r[0]:.4f}" for r in top[1:]))
    del x, z, src, out
    torch.cuda.empty_cache()

print()
for k, v in BEST.items(): print(k, v)
