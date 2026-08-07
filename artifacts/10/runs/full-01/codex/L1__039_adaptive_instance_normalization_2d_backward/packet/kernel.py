import torch
import triton
import triton.language as tl


@triton.jit
def _make_core_terms(
    grad_output, x, weight, mean, terms, partial_go,
    NUMEL: tl.constexpr, SPATIAL: tl.constexpr,
    NCHUNKS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = pid // NCHUNKS
    chunk = pid - nc * NCHUNKS
    local = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = local < SPATIAL
    offs = nc * SPATIAL + local
    go = tl.load(grad_output + offs, mask=mask, other=0.0)
    xv = tl.load(x + offs, mask=mask, other=0.0)
    mu = tl.load(mean + nc)
    gamma = tl.load(weight + (nc % 32))
    centered = tl.where(mask, xv - mu, 0.0)
    tl.store(terms + offs, (go * gamma) * centered, mask=mask)
    tl.store(terms + NUMEL + offs, (-2.0) * centered, mask=mask)
    tl.store(partial_go + nc * NCHUNKS + chunk, tl.sum(go, axis=0))


@triton.jit
def _make_terms(
    grad_output, x, weight, mean, std, terms,
    NUMEL: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    nc = offs // SPATIAL
    c = nc % 32
    go = tl.load(grad_output + offs, mask=mask)
    xv = tl.load(x + offs, mask=mask)
    mu = tl.load(mean + nc, mask=mask)
    sigma = tl.load(std + nc, mask=mask)
    gamma = tl.load(weight + c, mask=mask)
    centered = xv - mu
    scaled = go * gamma
    tl.store(terms + offs, go * (centered / sigma), mask=mask)
    tl.store(terms + NUMEL + offs, scaled * centered, mask=mask)
    tl.store(terms + 2 * NUMEL + offs, (-2.0) * centered, mask=mask)


@triton.jit
def _finish_exact_reductions(
    sum_var, sum_mean, sum_center, std, stats,
    NC: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr,
):
    nc = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = nc < NC
    sv = tl.load(sum_var + nc, mask=mask)
    sm = tl.load(sum_mean + nc, mask=mask)
    sc = tl.load(sum_center + nc, mask=mask)
    sigma = tl.load(std + nc, mask=mask)
    grad_var = sv * (-0.5) * tl.extra.libdevice.pow(sigma, -3.0)
    grad_mean = sm + grad_var * sc / SPATIAL
    tl.store(stats + nc, grad_var, mask=mask)
    tl.store(stats + NC + nc, grad_mean, mask=mask)


@triton.jit
def _finish_coefficients(
    sum_var, sum_go, weight, std, pow_std, sum_center,
    grad_var, grad_mean,
    NC: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr,
):
    nc = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = nc < NC
    channel = nc % 32
    sv = tl.load(sum_var + nc, mask=mask)
    sg = tl.load(sum_go + nc, mask=mask)
    gamma = tl.load(weight + channel, mask=mask)
    sigma = tl.load(std + nc, mask=mask)
    sigma_pow = tl.load(pow_std + nc, mask=mask)
    sc = tl.load(sum_center + nc, mask=mask)
    mean_first = (sg * gamma) / (-sigma)
    gv = (sv * (-0.5)) * sigma_pow
    tl.store(grad_var + nc, gv, mask=mask)
    tl.store(grad_mean + nc, mean_first, mask=mask)


@triton.jit
def _partial_reduce(
    grad_output, x, weight, mean, std, partial,
    SPATIAL: tl.constexpr, NCHUNKS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = pid // NCHUNKS
    chunk = pid - nc * NCHUNKS
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SPATIAL
    base = nc * SPATIAL + offs

    go = tl.load(grad_output + base, mask=mask, other=0.0)
    xv = tl.load(x + base, mask=mask, other=0.0)
    mu = tl.load(mean + nc)
    sigma = tl.load(std + nc)
    w = tl.load(weight + (nc % 32))

    centered = tl.where(mask, xv - mu, 0.0)
    normalized = centered / sigma
    scaled = go * w

    s_bias = tl.sum(go, axis=0)
    s_weight = tl.sum(go * normalized, axis=0)
    s_var = tl.sum(scaled * centered, axis=0)
    s_mean = tl.sum(scaled / (-sigma), axis=0)
    s_center = tl.sum((-2.0) * centered, axis=0)

    p = nc * NCHUNKS + chunk
    plane = tl.num_programs(0)
    tl.store(partial + p, s_bias)
    tl.store(partial + plane + p, s_weight)
    tl.store(partial + 2 * plane + p, s_var)
    tl.store(partial + 3 * plane + p, s_mean)
    tl.store(partial + 4 * plane + p, s_center)


@triton.jit
def _finish_instance(
    partial, std, stats,
    SPATIAL: tl.constexpr, NCHUNKS: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    offs = tl.arange(0, REDUCE_BLOCK)
    mask = offs < NCHUNKS
    p = nc * NCHUNKS + offs
    plane = tl.num_programs(0) * NCHUNKS

    s_bias = tl.sum(tl.load(partial + p, mask=mask, other=0.0), axis=0)
    s_weight = tl.sum(tl.load(partial + plane + p, mask=mask, other=0.0), axis=0)
    s_var = tl.sum(tl.load(partial + 2 * plane + p, mask=mask, other=0.0), axis=0)
    s_mean = tl.sum(tl.load(partial + 3 * plane + p, mask=mask, other=0.0), axis=0)
    s_center = tl.sum(tl.load(partial + 4 * plane + p, mask=mask, other=0.0), axis=0)

    sigma = tl.load(std + nc)
    grad_var = s_var * (-0.5) * tl.extra.libdevice.pow(sigma, -3.0)
    grad_mean = s_mean + grad_var * s_center / SPATIAL
    total_nc = tl.num_programs(0)
    tl.store(stats + nc, grad_var)
    tl.store(stats + total_nc + nc, grad_mean)
    tl.store(stats + 2 * total_nc + nc, s_weight)
    tl.store(stats + 3 * total_nc + nc, s_bias)


@triton.jit
def _write_grad_input(
    grad_output, x, weight, mean, std, grad_var_ptr, grad_mean_ptr, grad_input,
    NUMEL: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    nc = offs // SPATIAL
    channel = nc % 32

    go = tl.load(grad_output + offs, mask=mask)
    xv = tl.load(x + offs, mask=mask)
    w = tl.load(weight + channel, mask=mask)
    mu = tl.load(mean + nc, mask=mask)
    sigma = tl.load(std + nc, mask=mask)
    grad_var = tl.load(grad_var_ptr + nc, mask=mask)
    grad_mean = tl.load(grad_mean_ptr + nc, mask=mask)

    centered = xv - mu
    out = (go * w) / sigma
    out = out + grad_var * 2.0 * centered / SPATIAL
    out = out + grad_mean / SPATIAL
    tl.store(grad_input + offs, out, mask=mask)


@triton.jit
def _finish_channels(stats, grad_weight, grad_bias, N: tl.constexpr,
                     NC: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    mask = n < N
    nc = n * 32 + c
    gw = tl.sum(tl.load(stats + 2 * NC + nc, mask=mask, other=0.0), axis=0)
    gb = tl.sum(tl.load(stats + 3 * NC + nc, mask=mask, other=0.0), axis=0)
    tl.store(grad_weight + c, gw)
    tl.store(grad_bias + c, gb)


def run(grad_output, x, weight, mean, std):
    n, c, h, w = x.shape
    spatial = h * w
    nc = n * c
    numel = nc * spatial

    terms = torch.empty((3, n, c, h, w), device=x.device, dtype=torch.float32)
    grad_input = torch.empty_like(x)
    _make_terms[(triton.cdiv(numel, 512),)](
        grad_output, x, weight, mean, std, terms,
        NUMEL=numel, SPATIAL=spatial, BLOCK=512,
        num_warps=4,
    )

    grad_weight = terms[0].sum(dim=(0, 2, 3))
    spatial_sums = terms[1:3].sum(dim=(3, 4))
    sum_var = spatial_sums[0]
    sum_center = spatial_sums[1]
    sum_go = grad_output.sum(dim=(2, 3))
    grad_bias = grad_output.sum(dim=(0, 2, 3))

    pow_std = torch.pow(std, -3)
    grad_var = torch.empty_like(std)
    grad_mean = torch.empty_like(std)
    _finish_coefficients[(triton.cdiv(nc, 256),)](
        sum_var, sum_go, weight, std, pow_std, sum_center,
        grad_var, grad_mean,
        NC=nc, SPATIAL=spatial, BLOCK=256,
        num_warps=4,
    )
    grad_mean = grad_mean + grad_var * sum_center.view(n, c, 1, 1) / spatial
    _write_grad_input[(triton.cdiv(numel, 1024),)](
        grad_output, x, weight, mean, std, grad_var, grad_mean, grad_input,
        NUMEL=numel, SPATIAL=spatial, BLOCK=1024,
        num_warps=4,
    )
    return grad_input, grad_weight, grad_bias
