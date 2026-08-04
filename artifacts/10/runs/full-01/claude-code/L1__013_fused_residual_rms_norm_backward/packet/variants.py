"""Local experiment bed: parameterized kernel + microbenchmarks.

Not part of the solution. Used to pick block/grid/reduction strategy.
"""
import torch, triton, itertools
import triton.language as tl


# ---------------------------------------------------------------- main kernel
@triton.jit
def _bwd(GO, NORM, RSTD, W, GH, GR, PART, GW,
         M, NPROG,
         N: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr, R1: tl.constexpr,
         MASK1: tl.constexpr, MODE: tl.constexpr):
    # MODE 0 = no grad_weight at all (bandwidth floor probe)
    # MODE 1 = direct atomic into GW
    # MODE 2 = plain store into PART[pid]
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
def _reduce(PART, GW, NPART, N: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BC + tl.arange(0, BC)
    mask = cols < N
    acc = tl.zeros([BC], tl.float32)
    for i in range(NPART):
        acc += tl.load(PART + i * N + cols, mask=mask, other=0.0)
    tl.store(GW + cols, acc, mask=mask)


def split(N):
    if N & (N - 1) == 0:
        b0 = N // 2
        return b0, N - b0, N - b0, False
    b0 = 1 << (N.bit_length() - 1)
    r1 = N - b0
    b1 = triton.next_power_of_2(r1)
    return b0, b1, r1, b1 != r1


def make_run(nprog_fn, mode, num_warps, num_stages=1, bc=256):
    def run(grad_output, x, normalized, rstd, weight):
        N = grad_output.shape[-1]
        M = grad_output.numel() // N
        gh = torch.empty_like(grad_output)
        gr = torch.empty_like(grad_output)
        b0, b1, r1, mask1 = split(N)
        nprog = nprog_fn(M)
        if mode == 2:
            part = torch.empty((nprog, N), device=grad_output.device, dtype=torch.float32)
            gw = torch.empty(N, device=grad_output.device, dtype=torch.float32)
        elif mode == 1:
            part = grad_output
            gw = torch.zeros(N, device=grad_output.device, dtype=torch.float32)
        else:
            part = grad_output
            gw = torch.empty(N, device=grad_output.device, dtype=torch.float32)
        _bwd[(nprog,)](grad_output, normalized, rstd, weight, gh, gr, part, gw,
                       M, nprog, N=N, B0=b0, B1=b1, R1=r1, MASK1=mask1, MODE=mode,
                       num_warps=num_warps, num_stages=num_stages)
        if mode == 2:
            _reduce[(triton.cdiv(N, bc),)](part, gw, nprog, N=N, BC=bc, num_warps=4)
        return gh, gr, gw
    return run


# ---------------------------------------------------------------- benchmarking
def bench_ev(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


DEV = "cuda:0"
HID = 2560


def mk(M):
    b, s = 1, M
    return (torch.randn(b, s, HID, device=DEV, dtype=torch.bfloat16),
            torch.randn(b, s, HID, device=DEV, dtype=torch.float32),
            torch.randn(b, s, HID, device=DEV, dtype=torch.float32),
            torch.randn(b, s, 1, device=DEV, dtype=torch.float32),
            torch.randn(HID, device=DEV, dtype=torch.float32))


if __name__ == "__main__":
    import sys
    SIZES = [128, 512, 1546, 8192, 34784, 131072, 524288]

    # empty-launch overhead reference
    @triton.jit
    def _nop(X):
        pass
    xdum = torch.empty(1, device=DEV)
    t = bench_ev(lambda: _nop[(1,)](xdum), ())
    print(f"empty triton launch: {t*1000:.2f} us\n")

    print("=== MODE 0 (no grad_weight) : bandwidth floor, vary num_warps ===")
    for M in SIZES:
        args = mk(M)
        line = f"M={M:7d}"
        for nw in (2, 4, 8):
            f = make_run(lambda m: min(m, 2048), 0, nw)
            t = bench_ev(f, args)
            gb = M * HID * 10 / (t * 1e-3) / 1e9
            line += f" | w{nw}: {t*1000:8.1f}us {gb:7.0f}GB/s"
        print(line)

    print("\n=== grad_weight strategy (num_warps=4) ===")
    for M in SIZES:
        args = mk(M)
        line = f"M={M:7d}"
        for name, mode in (("none", 0), ("atomic", 1), ("part", 2)):
            f = make_run(lambda m: min(m, 2048), mode, 4)
            t = bench_ev(f, args)
            line += f" | {name}: {t*1000:8.1f}us"
        print(line)

    print("\n=== NPROG sweep, mode=part, num_warps=4 ===")
    for M in SIZES:
        args = mk(M)
        line = f"M={M:7d}"
        for np_ in (256, 512, 1024, 2048, 4096):
            f = make_run(lambda m, k=np_: min(m, k), 2, 4)
            t = bench_ev(f, args)
            line += f" | {np_}: {t*1000:7.1f}us"
        print(line)

    print("\n=== NPROG sweep, mode=atomic, num_warps=4 ===")
    for M in SIZES:
        args = mk(M)
        line = f"M={M:7d}"
        for np_ in (256, 512, 1024, 2048, 4096):
            f = make_run(lambda m, k=np_: min(m, k), 1, 4)
            t = bench_ev(f, args)
            line += f" | {np_}: {t*1000:7.1f}us"
        print(line)
