import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xb, stride_xh,
    stride_yb, stride_yh,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    # Compute mean(x^2) across the row of length N.
    # Load the whole row (N == 7168) in BLOCK chunks.
    acc = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xb + cols * stride_xh, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    mean = acc / N
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    # Second pass: normalize and scale.
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xb + cols * stride_xh, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * inv_rms) * w
        tl.store(Y_ptr + row * stride_yb + cols * stride_yh, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    x = hidden_states.contiguous()
    y = torch.empty_like(hidden_states)
    w = weight.contiguous()
    BLOCK = 1024
    _rmsnorm_fwd_kernel[(batch_size,)](
        x, w, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N=hidden_size, BLOCK=BLOCK, EPS=1e-6,
        num_warps=8,
    )
    return y
