import torch
import triton
import triton.language as tl


@triton.jit
def _gn_stats(X, PSUM, PSQ, N, NSPLIT, MEAN, HAVE_MEAN: tl.constexpr,
              BLOCK: tl.constexpr):
    pid_g = tl.program_id(0)
    pid_s = tl.program_id(1)
    base = pid_g.to(tl.int64) * N
    per = (N + NSPLIT - 1) // NSPLIT
    start = pid_s * per
    end = tl.minimum(start + per, N)
    mu = tl.zeros([], dtype=tl.float32)
    if HAVE_MEAN:
        mu = tl.load(MEAN + pid_g)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        m = idx < end
        v = tl.load(X + base + idx, mask=m, other=0.0)
        if HAVE_MEAN:
            d = tl.where(m, v - mu, 0.0)
            acc2 += d * d
        else:
            acc += v
        off += BLOCK
    if HAVE_MEAN:
        tl.store(PSQ + pid_g * NSPLIT + pid_s, tl.sum(acc2, axis=0))
    else:
        tl.store(PSUM + pid_g * NSPLIT + pid_s, tl.sum(acc, axis=0))


@triton.jit
def _gn_mean(PSUM, MEAN, N, NSPLIT, NS_POW2: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, NS_POW2)
    m = idx < NSPLIT
    s = tl.sum(tl.load(PSUM + pid * NSPLIT + idx, mask=m, other=0.0), axis=0)
    tl.store(MEAN + pid, s / N.to(tl.float32))


@triton.jit
def _gn_rstd(PSQ, RSTD, N, NSPLIT, eps, NS_POW2: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, NS_POW2)
    m = idx < NSPLIT
    s2 = tl.sum(tl.load(PSQ + pid * NSPLIT + idx, mask=m, other=0.0), axis=0)
    var = tl.maximum(s2 / N.to(tl.float32), 0.0)
    tl.store(RSTD + pid, 1.0 / tl.sqrt(var + eps))


@triton.jit
def _gn_apply(X, Y, MEAN, RSTD, GW, GB, S, C, CG, SILU: tl.constexpr, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)   # b*C + c
    pid_s = tl.program_id(1)
    c = pid_r % C
    b = pid_r // C
    g = b * (C // CG) + c // CG
    mean = tl.load(MEAN + g)
    rstd = tl.load(RSTD + g)
    w = tl.load(GW + c)
    bb = tl.load(GB + c)
    a = rstd * w
    sh = bb - mean * a
    idx = pid_s * BLOCK + tl.arange(0, BLOCK)
    m = idx < S
    p = pid_r.to(tl.int64) * S + idx
    v = tl.load(X + p, mask=m, other=0.0)
    y = a * v + sh
    if SILU:
        y = y / (1.0 + tl.exp(-y))
    tl.store(Y + p, y, mask=m)


def gn_silu(x, num_groups, weight, bias, eps, silu=True, out=None):
    """x: (B, C, ...) contiguous float32."""
    B, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    CG = C // num_groups
    N = CG * S
    NG = B * num_groups
    dev = x.device
    nsplit = max(1, min(1024, triton.cdiv(N, 32768)))
    nsp2 = triton.next_power_of_2(nsplit)
    ps = torch.empty((NG, nsplit), device=dev, dtype=torch.float32)
    mean = torch.empty(NG, device=dev, dtype=torch.float32)
    rstd = torch.empty(NG, device=dev, dtype=torch.float32)
    _gn_stats[(NG, nsplit)](x, ps, ps, N, nsplit, mean, HAVE_MEAN=False,
                            BLOCK=2048, num_warps=8)
    _gn_mean[(NG,)](ps, mean, N, nsplit, NS_POW2=nsp2)
    _gn_stats[(NG, nsplit)](x, ps, ps, N, nsplit, mean, HAVE_MEAN=True,
                            BLOCK=2048, num_warps=8)
    _gn_rstd[(NG,)](ps, rstd, N, nsplit, eps, NS_POW2=nsp2)
    y = torch.empty_like(x) if out is None else out
    BLK = 2048
    _gn_apply[(B * C, triton.cdiv(S, BLK))](
        x, y, mean, rstd, weight, bias, S, C, CG, SILU=silu, BLOCK=BLK, num_warps=8)
    return y
