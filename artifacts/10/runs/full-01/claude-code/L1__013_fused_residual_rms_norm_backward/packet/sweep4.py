import torch, triton
import triton.language as tl
from variants import bench_ev, mk, DEV, HID, split

# main kernel, MODE: 0 none, 1 atomic, 2 partials
@triton.jit
def _bwd(GO, NORM, RSTD, W, GH, GR, PART, GW, M, NPROG,
         N: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
         MASK1: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    INV = 1.0 / N
    c0 = tl.arange(0, B0)
    w0 = tl.load(W + c0)
    a0 = tl.zeros([B0], tl.float32)
    c1 = B0 + tl.arange(0, B1)
    if MASK1:
        m1 = c1 < N
        w1 = tl.load(W + c1, mask=m1, other=0.0)
    else:
        w1 = tl.load(W + c1)
    a1 = tl.zeros([B1], tl.float32)

    row = pid
    while row < M:
        base = row.to(tl.int64) * N
        g0 = tl.load(GO + base + c0).to(tl.float32)
        n0 = tl.load(NORM + base + c0)
        if MASK1:
            g1 = tl.load(GO + base + c1, mask=m1, other=0.0).to(tl.float32)
            n1 = tl.load(NORM + base + c1, mask=m1, other=0.0)
        else:
            g1 = tl.load(GO + base + c1).to(tl.float32)
            n1 = tl.load(NORM + base + c1)
        rs = tl.load(RSTD + row)
        p0 = g0 * n0
        p1 = g1 * n1
        if MODE != 0:
            a0 += p0
            a1 += p1
        s = (tl.sum(w0 * p0, axis=0) + tl.sum(w1 * p1, axis=0)) * INV
        o0 = (rs * (g0 * w0 - s * n0)).to(tl.bfloat16)
        o1 = (rs * (g1 * w1 - s * n1)).to(tl.bfloat16)
        tl.store(GH + base + c0, o0)
        tl.store(GR + base + c0, o0)
        if MASK1:
            tl.store(GH + base + c1, o1, mask=m1)
            tl.store(GR + base + c1, o1, mask=m1)
        else:
            tl.store(GH + base + c1, o1)
            tl.store(GR + base + c1, o1)
        row += NPROG

    if MODE == 1:
        tl.atomic_add(GW + c0, a0)
        if MASK1:
            tl.atomic_add(GW + c1, a1, mask=m1)
        else:
            tl.atomic_add(GW + c1, a1)
    elif MODE == 2:
        pb = pid.to(tl.int64) * N
        tl.store(PART + pb + c0, a0)
        if MASK1:
            tl.store(PART + pb + c1, a1, mask=m1)
        else:
            tl.store(PART + pb + c1, a1)


@triton.jit
def _red2(PART, TMP, P, N, SPL: tl.constexpr, BC: tl.constexpr):
    cb = tl.program_id(0); sb = tl.program_id(1)
    cols = cb * BC + tl.arange(0, BC)
    m = cols < N
    acc = tl.zeros([BC], tl.float32)
    i = sb
    while i < P:
        acc += tl.load(PART + i * N + cols, mask=m, other=0.0)
        i += SPL
    tl.store(TMP + sb * N + cols, acc, mask=m)


@triton.jit
def _red1(PART, GW, P: tl.constexpr, N, BC: tl.constexpr):
    cb = tl.program_id(0)
    cols = cb * BC + tl.arange(0, BC)
    m = cols < N
    acc = tl.zeros([BC], tl.float32)
    for i in range(P):
        acc += tl.load(PART + i * N + cols, mask=m, other=0.0)
    tl.store(GW + cols, acc, mask=m)


def make(mode, nprog_fn, nw, spl=32, bc=128):
    def run(go, x, nm, rstd, w):
        N = go.shape[-1]; M = go.numel() // N
        gh = torch.empty_like(go); gr = torch.empty_like(go)
        b0, b1, r1, mask1 = split(N)
        P = nprog_fn(M)
        if mode == 2:
            part = torch.empty((P, N), device=go.device, dtype=torch.float32)
            gw = torch.empty(N, device=go.device, dtype=torch.float32)
        elif mode == 1:
            part = go; gw = torch.zeros(N, device=go.device, dtype=torch.float32)
        else:
            part = go; gw = torch.empty(N, device=go.device, dtype=torch.float32)
        _bwd[(P,)](go, nm, rstd, w, gh, gr, part, gw, M, P,
                   N=N, B0=b0, B1=b1, MASK1=mask1, MODE=mode,
                   num_warps=nw, num_stages=1)
        if mode == 2:
            s = min(spl, P)
            if s <= 1:
                _red1[(triton.cdiv(N, bc),)](part, gw, P, N, BC=bc, num_warps=4)
            else:
                tmp = torch.empty((s, N), device=go.device, dtype=torch.float32)
                _red2[(triton.cdiv(N, bc), s)](part, tmp, P, N, SPL=s, BC=bc, num_warps=4)
                gw = tmp.sum(0)
        return gh, gr, gw
    return run


MS = [128, 422, 1249, 1546, 3412, 8192, 15952, 34784, 53024, 110144, 131072, 524288]
NPS = [128, 256, 512, 1024, 2048]

if __name__ == "__main__":
    print("=== mode0 (no gw) vs mode1 (atomic) vs mode2 (partial+fast reduce), num_warps=4 ===")
    for M in MS:
        args = mk(M)
        print(f"M={M}")
        for P in NPS:
            if P > 4 * M:
                continue
            line = f"   nprog={P:5d}:"
            for nm_, md in (("none", 0), ("atom", 1), ("part", 2)):
                f = make(md, lambda m, k=P: min(m, k), 4)
                t = bench_ev(f, args)
                line += f"  {nm_}={t*1000:8.1f}us"
            print(line)
        del args; torch.cuda.empty_cache()
