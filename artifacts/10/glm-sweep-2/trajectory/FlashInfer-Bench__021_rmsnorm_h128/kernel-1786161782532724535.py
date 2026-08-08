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
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_R

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_r = tl.arange(0, BLOCK_R)
    rows = row_start + offs_r
    mask_r = rows < N_ROWS

    w_row = tl.load(w_ptr + offs_h * stride_wr, mask=mask_h, other=0.0).to(tl.float32)

    x = tl.load(
        x_ptr + rows[:, None] * stride_xr + offs_h[None, :] * stride_xc,
        mask=mask_r[:, None] & mask_h[None, :],
        other=0.0,
    ).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=1) / H
    inv_rms = tl.rsqrt(mean_sq + EPS)
    y = (x * inv_rms[:, None]) * w_row[None, :]

    tl.store(
        y_ptr + rows[:, None] * stride_yr + offs_h[None, :] * stride_yc,
        y.to(tl.bfloat16),
        mask=mask_r[:, None] & mask_h[None, :],
    )


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
    BLOCK_R = 8
    grid = (triton.cdiv(batch_size, BLOCK_R),)
    _rmsnorm_fwd_kernel[grid](
        x, w, y,
        batch_size, hidden_size,
        x.stride(0), x.stride(1),
        w.stride(0),
        y.stride(0), y.stride(1),
        EPS=EPS,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        num_warps=2,
        num_stages=4,
    )
    return y
