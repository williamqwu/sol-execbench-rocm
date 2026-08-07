import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rope_backward_kernel(
    grad_cos,
    grad_sin,
    idx_theta,
    output,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(idx_theta + offsets, mask=mask)
    gc = tl.load(grad_cos + offsets, mask=mask).to(tl.float32)
    gs = tl.load(grad_sin + offsets, mask=mask).to(tl.float32)
    result = (-gc) * tl.sin(x) + gs * tl.cos(x)
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _rope_backward_sincos_kernel(
    grad_cos,
    grad_sin,
    idx_theta,
    output,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN_N: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        mask = None
    else:
        mask = offsets < n_elements

    x = tl.load(idx_theta + offsets, mask=mask, cache_modifier=LOAD_CACHE)
    gc = tl.load(grad_cos + offsets, mask=mask, cache_modifier=LOAD_CACHE).to(tl.float32)
    gs = tl.load(grad_sin + offsets, mask=mask, cache_modifier=LOAD_CACHE).to(tl.float32)

    # The gfx950 OCML sin/cos path uses the same quadrant reduction for both
    # functions.  Doing it once here avoids duplicating that work.
    ax = tl.abs(x)
    quadrant_f = libdevice.rint(ax * 0.6366197466850281)
    reduced = tl.fma(quadrant_f, -1.570796251296997, ax)
    reduced = tl.fma(quadrant_f, -7.549789415861596e-08, reduced)
    reduced = tl.fma(quadrant_f, -5.390302529957765e-15, reduced)
    quadrant = quadrant_f.to(tl.int32) & 3

    z = reduced * reduced
    sin_p = tl.fma(z, -0.00019464458455331624, 0.008331719785928726)
    sin_p = tl.fma(z, sin_p, -0.16666646301746368)
    sin_p = z * sin_p
    sin_r = tl.fma(reduced, sin_p, reduced)

    cos_p = tl.fma(z, 2.5668741727713495e-05, -0.0013909110566601157)
    cos_p = tl.fma(z, cos_p, 0.04166790470480919)
    cos_p = tl.fma(z, cos_p, -0.5000002384185791)
    cos_r = tl.fma(z, cos_p, 1.0)

    even = (quadrant & 1) == 0
    upper_half = quadrant > 1
    sin_x = tl.where(even, sin_r, cos_r)
    sin_x = tl.where(upper_half, -sin_x, sin_x)
    sin_x = tl.where(x < 0.0, -sin_x, sin_x)
    cos_x = tl.where(even, cos_r, -sin_r)
    cos_x = tl.where(upper_half, -cos_x, cos_x)

    result = (-gc) * sin_x + gs * cos_x
    tl.store(output + offsets, result, mask=mask, cache_modifier=STORE_CACHE)


@triton.jit
def _rope_backward_persistent_kernel(
    grad_cos,
    grad_sin,
    idx_theta,
    output,
    n_elements,
    BLOCK: tl.constexpr,
):
    first = tl.program_id(0) * BLOCK
    stride = tl.num_programs(0) * BLOCK
    for block_start in range(first, n_elements, stride):
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(idx_theta + offsets, mask=mask)
        gc = tl.load(grad_cos + offsets, mask=mask).to(tl.float32)
        gs = tl.load(grad_sin + offsets, mask=mask).to(tl.float32)
        result = (-gc) * tl.sin(x) + gs * tl.cos(x)
        tl.store(output + offsets, result, mask=mask)


def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(idx_theta)
    n_elements = idx_theta.numel()
    if n_elements < 16_000_000:
        block = 256
        num_warps = 1
        waves_per_eu = 7
    else:
        block = 2048
        num_warps = 8
        waves_per_eu = 0
    _rope_backward_sincos_kernel[(triton.cdiv(n_elements, block),)](
        grad_cos,
        grad_sin,
        idx_theta,
        output,
        n_elements=n_elements,
        BLOCK=block,
        EVEN_N=(n_elements % block == 0),
        LOAD_CACHE=".cg",
        STORE_CACHE=".wt",
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return output
