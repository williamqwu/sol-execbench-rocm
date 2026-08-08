import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xb, stride_yb,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_xb + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    y = x * (inv_rms * w)
    tl.store(Y_ptr + row * stride_yb + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    y = torch.empty_like(hidden_states)
    _rmsnorm_fwd_kernel[(batch_size,)](
        hidden_states, weight, y,
        hidden_states.stride(0), y.stride(0),
        N=hidden_size, BLOCK=8192, EPS=1e-6,
        num_warps=8,
    )
    return y
