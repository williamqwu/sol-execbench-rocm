import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x_row, stride_y_row,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * stride_x_row
    y_row_ptr = y_ptr + row * stride_y_row

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # compute mean(x^2)
    x2 = x * x
    mean_x2 = tl.sum(x2) / N
    inv_rms = tl.rsqrt(mean_x2 + EPS)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 1536
    x = hidden_states
    assert x.is_cuda and x.dtype == torch.bfloat16

    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(hidden_size)
    num_warps = 4
    _rmsnorm_kernel[(batch_size,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        N=hidden_size,
        BLOCK_N=BLOCK_N,
        EPS=1e-6,
        num_warps=num_warps,
    )
    return y
