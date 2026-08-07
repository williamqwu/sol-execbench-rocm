import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rope_kernel(
    ids,
    output,
    n_rows: tl.constexpr,
    log2_theta,
):
    row = tl.program_id(0)
    band = tl.arange(0, 64)

    # The three half-dimensions are 8, 28, and 28.  Their doubled,
    # interleaved destinations happen to be simply 2 * band throughout.
    exponent = tl.where(band < 8, band, tl.where(band < 36, band - 8, band - 36))
    half_dim = tl.where(band < 8, 8.0, 28.0)

    p0 = tl.load(ids + row * 3)
    p1 = tl.load(ids + row * 3 + 1)
    p2 = tl.load(ids + row * 3 + 2)
    position = tl.where(band < 8, p0, tl.where(band < 36, p1, p2))

    power = exponent.to(tl.float32) / half_dim
    freq = tl.exp2(-power * log2_theta)
    angle = position * freq
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)

    out_col = tl.arange(0, 128)
    dst = row * 128 + out_col
    tl.store(output + dst, tl.interleave(c, c))
    tl.store(output + n_rows * 128 + dst, tl.interleave(s, s))


def run(ids: torch.Tensor, theta: float):
    n_rows = ids.shape[0]
    output = torch.empty((2, n_rows, 128), device=ids.device, dtype=torch.float32)
    _rope_kernel[(n_rows,)](
        ids,
        output,
        n_rows=n_rows,
        log2_theta=math.log2(theta),
        num_warps=1,
    )
    return output[0], output[1]
