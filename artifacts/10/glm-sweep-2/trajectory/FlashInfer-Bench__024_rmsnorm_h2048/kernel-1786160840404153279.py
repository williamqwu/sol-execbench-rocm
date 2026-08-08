import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_row,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Compute mean(x^2) across the row of length H
    # Block-based reduction: load the whole row (H=2048) in one block
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    # mean of squares
    sq = x * x
    mean_sq = tl.sum(sq) / H
    inv_rms = tl.rsqrt(mean_sq + 1e-6)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 2048

    out = torch.empty_like(hidden_states)
    BLOCK = 2048
    num_warps = 8
    _rmsnorm_kernel[(batch_size,)](
        hidden_states, weight, out,
        hidden_states.stride(0),
        H=hidden_size,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
