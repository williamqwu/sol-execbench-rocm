import torch
import triton
import triton.language as tl
import math

_SQRT_H = math.sqrt(512)
_EPS_H = 1e-6 * 512


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    SQRT_H: tl.constexpr,
    EPS_H: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    # Load weight first so it's in-flight during reduction
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    # Fold /H into rsqrt: inv_rms = sqrt(H) * rsqrt(sum_sq + EPS*H)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = SQRT_H * tl.rsqrt(sum_sq + EPS_H)
    y = (x * inv_rms) * w
    tl.store(y_ptr + row * H + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 512
    y = torch.empty_like(hidden_states)
    _rmsnorm_kernel[(B,)](
        hidden_states, weight, y,
        SQRT_H=_SQRT_H, EPS_H=_EPS_H, H=H, BLOCK=512,
        num_warps=1,
    )
    return y
