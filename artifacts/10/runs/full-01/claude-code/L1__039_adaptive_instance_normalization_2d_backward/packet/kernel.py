import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# Pass 1: per-(n,c) partial reductions over the spatial dimension.
#
# For every instance (n, c) with S = H*W we need
#   A = sum(go)                      -> grad_bias
#   B = sum(go * ((x-mean)/std))     -> grad_weight
#   Cc= sum((go*w) * (x-mean))       -> grad_var
#   D = sum((go*w) / (-std))         -> grad_mean (part 1)
#   E = sum(-2*(x-mean))             -> grad_mean (part 2)
# ---------------------------------------------------------------------------
@triton.jit
def _planes_kernel(
    GO, X, W, MEAN, STD, PL, PROD,
    S, C, NCS,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    blk = tl.program_id(1)
    c = nc % C

    w = tl.load(W + c)
    mean = tl.load(MEAN + nc)
    std = tl.load(STD + nc)

    base = nc.to(tl.int64) * S
    o = blk * BLOCK + tl.arange(0, BLOCK)
    m = o < S
    p = base + o

    go = tl.load(GO + p, mask=m, other=0.0)
    x = tl.load(X + p, mask=m, other=0.0)

    xc = x - mean
    gos = go * w

    tl.store(PL + 0 * NCS + p, gos * xc, mask=m)
    tl.store(PL + 1 * NCS + p, tl.div_rn(gos, -std), mask=m)
    tl.store(PL + 2 * NCS + p, -2.0 * xc, mask=m)
    tl.store(PROD + p, go * tl.div_rn(xc, std), mask=m)


# ---------------------------------------------------------------------------
# Pass 3: elementwise grad_input.
# ---------------------------------------------------------------------------
@triton.jit
def _elem_kernel(
    GO, X, W, MEAN, STD, K1S, K2, GI,
    S, C, S_F,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    blk = tl.program_id(1)
    c = nc % C

    w = tl.load(W + c)
    mean = tl.load(MEAN + nc)
    std = tl.load(STD + nc)
    k1s = tl.load(K1S + nc)
    k2 = tl.load(K2 + nc)

    base = nc.to(tl.int64) * S
    o = blk * BLOCK + tl.arange(0, BLOCK)
    m = o < S

    go = tl.load(GO + base + o, mask=m, other=0.0)
    x = tl.load(X + base + o, mask=m, other=0.0)

    xc = x - mean
    gi = tl.div_rn(go * w, std)
    gi = gi + tl.div_rn(k1s * xc, S_F)
    gi = gi + k2
    tl.store(GI + base + o, gi, mask=m)


_STREAMS = {}


def _streams(dev):
    st = _STREAMS.get(dev.index)
    if st is None:
        st = (torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev))
        _STREAMS[dev.index] = st
    return st


def _floor_pow2(v):
    v = int(v)
    if v < 1:
        return 1
    return 1 << (v.bit_length() - 1)


@torch.no_grad()
def run(grad_output, x, weight, mean, std):
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    mean = mean.contiguous()
    std = std.contiguous()

    N, C, H, W = x.shape
    S = H * W
    NC = N * C
    NCS = NC * S

    dev = x.device
    grad_input = torch.empty_like(x)

    if S == 0 or NC == 0:
        z = torch.zeros(C, device=dev, dtype=torch.float32)
        return grad_input, z, z.clone()

    # One pass over the data emits the three per-instance product planes plus
    # the grad_weight integrand. The reductions themselves must stay in torch:
    # the workload tolerances are near fp32 epsilon and tighter than the
    # reduction's own rounding error, so any re-association -- even a strictly
    # more accurate one -- lands outside them. A batched sum(dim=(3,4)) over
    # stacked planes is bit-identical to reducing each plane on its own.
    planes = torch.empty(3, N, C, H, W, device=dev, dtype=torch.float32)
    prod = torch.empty_like(x)

    BLOCK_P = 1024
    _planes_kernel[(NC, triton.cdiv(S, BLOCK_P))](
        grad_output, x, weight, mean, std, planes, prod,
        S, C, NCS,
        BLOCK=BLOCK_P,
        num_warps=4,
        num_stages=2,
    )

    overlap = NCS >= (1 << 20)
    main = torch.cuda.current_stream(dev)

    if overlap:
        sb, sw = _streams(dev)
        sb.wait_stream(main)
        sw.wait_stream(main)
        with torch.cuda.stream(sb):
            grad_output.record_stream(sb)
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        with torch.cuda.stream(sw):
            prod.record_stream(sw)
            grad_weight = prod.sum(dim=(0, 2, 3))
        R = planes.sum(dim=(3, 4))
        main.wait_stream(sb)
        main.wait_stream(sw)
    else:
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        grad_weight = prod.sum(dim=(0, 2, 3))
        R = planes.sum(dim=(3, 4))

    # The per-instance algebra stays in torch: it is only N*C elements, and
    # torch.pow(std, -3) has no bit-exact Triton equivalent (libdevice.pow is
    # ~1e-6 off, which std^-3 amplifies straight past the tolerance).
    cc = R[0].view(N, C, 1, 1)
    d = R[1].view(N, C, 1, 1)
    e = R[2].view(N, C, 1, 1)
    grad_var = cc * (-0.5) * torch.pow(std, -3)
    grad_mean = d + grad_var * e / S
    k1s = (grad_var * 2.0).reshape(-1)
    k2 = (grad_mean / S).reshape(-1)

    BLOCK_E = 2048
    _elem_kernel[(NC, triton.cdiv(S, BLOCK_E))](
        grad_output, x, weight, mean, std, k1s, k2, grad_input,
        S, C, float(S),
        BLOCK=BLOCK_E,
        num_warps=8,
        num_stages=2,
    )

    return grad_input, grad_weight, grad_bias
