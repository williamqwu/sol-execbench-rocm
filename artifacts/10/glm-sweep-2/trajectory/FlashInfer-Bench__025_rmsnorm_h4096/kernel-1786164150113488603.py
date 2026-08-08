import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean + EPS)

    y = x * inv_rms * w
    tl.store(Y_ptr + row * stride_x + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096

    y = torch.empty_like(hidden_states)

    _rmsnorm_kernel[(batch_size,)](
        hidden_states, weight, y,
        hidden_states.stride(0),
        N=4096,
        EPS=1e-5,
        BLOCK_N=4096,
        num_warps=4,
        num_stages=3,
    )
    return y
