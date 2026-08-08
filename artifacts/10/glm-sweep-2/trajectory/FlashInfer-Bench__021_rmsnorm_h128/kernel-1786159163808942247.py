import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr,
    N_ROWS, H,
    stride_xr, stride_xc,
    stride_wr,
    stride_yr, stride_yc,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N_ROWS:
        return

    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x_row = tl.load(x_ptr + row * stride_xr + offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
    w_row = tl.load(w_ptr + offs * stride_wr, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x_row * x_row, axis=0) / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    y = (x_row * inv_rms) * w_row
    tl.store(y_ptr + row * stride_yr + offs * stride_yc, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 128
    assert hidden_states.is_cuda and weight.is_cuda
    assert hidden_states.dtype == torch.bfloat16

    EPS = 1e-6
    x = hidden_states.contiguous()
    w = weight.contiguous()
    y = torch.empty_like(x)

    BLOCK_H = triton.next_power_of_2(hidden_size)  # 128
    grid = (batch_size,)
    _rmsnorm_fwd_kernel[grid](
        x, w, y,
        batch_size, hidden_size,
        x.stride(0), x.stride(1),
        w.stride(0),
        y.stride(0), y.stride(1),
        EPS=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=2,
    )
    return y
