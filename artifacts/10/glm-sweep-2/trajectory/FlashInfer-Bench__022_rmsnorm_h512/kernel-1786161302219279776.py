import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    B,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    wblk = tl.arange(0, WARPS)
    cols = tl.arange(0, BLOCK)
    row_idx = pid * WARPS + wblk
    row_mask = row_idx < B
    col_mask = cols < H

    x_ptrs = x_ptr + row_idx[:, None] * H + cols[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=1) / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = (x * inv_rms[:, None]) * w[None, :]
    y_ptrs = y_ptr + row_idx[:, None] * H + cols[None, :]
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 512
    y = torch.empty_like(hidden_states)
    warps = 1
    _rmsnorm_kernel[(triton.cdiv(B, warps),)](
        hidden_states, weight, y,
        B, H=H, EPS=1e-6, BLOCK=512, WARPS=warps,
        num_warps=warps,
    )
    return y
