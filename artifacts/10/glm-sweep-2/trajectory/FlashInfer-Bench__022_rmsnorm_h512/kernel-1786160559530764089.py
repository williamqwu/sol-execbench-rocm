import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    # Reduction in fp32
    mean_sq = tl.sum(x * x, axis=0) / H
    inv_rms = tl.rsqrt(mean_sq + EPS)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(y_ptr + row * H + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 512
    y = torch.empty_like(hidden_states)
    _rmsnorm_kernel[(B,)](
        hidden_states, weight, y,
        H=H, EPS=1e-6, BLOCK=512,
        num_warps=1,
    )
    return y
