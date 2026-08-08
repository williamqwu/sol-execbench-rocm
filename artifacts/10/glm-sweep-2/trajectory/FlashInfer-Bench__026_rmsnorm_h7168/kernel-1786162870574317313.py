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
    x2 = tl.where(mask, x * x, 0.0)
    mean = tl.sum(x2, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + row * stride_yb + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    x = hidden_states.contiguous()
    y = torch.empty_like(hidden_states)
    w = weight.contiguous()
    # 7168 elements; round up to power of two for Triton.
    BLOCK = 8192
    _rmsnorm_fwd_kernel[(batch_size,)](
        x, w, y,
        x.stride(0), y.stride(0),
        N=hidden_size, BLOCK=BLOCK, EPS=1e-6,
        num_warps=4,
    )
    return y
