import torch, triton, triton.language as tl, time, json
dev = 'cuda:0'
RTOL = 1.1920928955078125e-07


# ---------------- variants ----------------
@triton.jit
def v1(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


@triton.jit
def v2(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid*BLOCK
    o2 = base*2 + tl.arange(0, 2*BLOCK)
    v = tl.load(SRC + o2, mask=o2 < 2*M, other=0.)
    re, im = tl.split(v.reshape(BLOCK, 2))
    offs = base + tl.arange(0, BLOCK); m = offs < M
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


@triton.jit
def v3(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    re = tl.load(SRC + 2*offs, mask=m, other=0.)
    im = tl.load(SRC + 2*offs + 1, mask=m, other=0.)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


def bench(f, iters=30, warm=10):
    for _ in range(warm): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]*1e3


rows = [json.loads(l) for l in open('workload.jsonl')]
print(f"{'B':>3}{'S':>7} {'ref':>7} {'rfft':>7} {'fwd':>7} | {'v1':>6} {'v2':>6} {'v3':>6} {'perm':>6} {'refsp':>6} | best_total")
for w in rows:
    B = w['axes']['batch_size']; S = w['axes']['seqlen']
    x = torch.randn(B, 256, S, device=dev, dtype=torch.float32)
    n = 2*S
    t_ref = bench(lambda: (lambda z: (z.real.contiguous(), z.imag.contiguous()))(torch.fft.rfft(x, n=n)/n))
    t_rfft = bench(lambda: torch.fft.rfft(x, n=n))
    t_fwd = bench(lambda: torch.fft.rfft(x, n=n, norm='forward'))
    z = torch.fft.rfft(x, n=n)
    flat = z.view(torch.float32).reshape(-1); M = flat.numel()//2
    out = torch.empty((2, B, 256, S+1), device=dev, dtype=torch.float32)
    re = out[0].reshape(-1); im = out[1].reshape(-1)
    inv = 1.0/n
    res = {}
    for nm, kern, bs, nw in [("v1", v1, 1024, 4), ("v2", v2, 1024, 4), ("v3", v3, 1024, 4)]:
        f = lambda kern=kern, bs=bs, nw=nw: kern[(triton.cdiv(M, bs),)](flat, re, im, M, inv, BLOCK=bs, num_warps=nw)
        res[nm] = bench(f)
    vr = torch.view_as_real(z)
    t_perm = bench(lambda: vr.permute(3, 0, 1, 2).contiguous())
    t_refsp = bench(lambda: ((z/n).real.contiguous(), (z/n).imag.contiguous()))
    best = min(res.values())
    print(f"{B:>3}{S:>7} {t_ref:7.3f} {t_rfft:7.3f} {t_fwd:7.3f} | {res['v1']:6.3f} {res['v2']:6.3f} {res['v3']:6.3f} {t_perm:6.3f} {t_refsp:6.3f} | {t_rfft+best:7.3f}")
    del x, z, out, flat
    torch.cuda.empty_cache()
