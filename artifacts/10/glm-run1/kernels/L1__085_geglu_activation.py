import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _geglu_kernel(x_ptr, y_ptr, num_rows, inner, BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr):
    pid = tl.program_id(0)
    n_col = tl.cdiv(inner, BLOCK_COL)
    pid_row = pid // n_col
    pid_col = pid % n_col
    row_offs = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = row_offs < num_rows
    col_offs = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = col_offs < inner
    xrp = x_ptr + row_offs[:, None] * (2 * inner)
    yrp = y_ptr + row_offs[:, None] * inner
    m = row_mask[:, None] & col_mask[None, :]
    g = tl.load(xrp + col_offs[None, :], mask=m, other=0.0).to(tl.float32)
    l = tl.load(xrp + col_offs[None, :] + inner, mask=m, other=0.0).to(tl.float32)
    SQRT_2_PI = 0.7978845608028654
    c = SQRT_2_PI * (g + 0.044715 * g * g * g)
    t = 2.0 * tl.sigmoid(2.0 * c) - 1.0
    gelu = 0.5 * g * (1.0 + t)
    tl.store(yrp + col_offs[None, :], gelu * l, mask=m)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    inner = x.shape[-1] // 2
    num_rows = x.numel() // (2 * inner)
    y = torch.empty(x.shape[:-1] + (inner,), dtype=x.dtype, device=x.device)
    BLOCK_COL = 256
    BLOCK_ROW = 1
    grid = (triton.cdiv(num_rows, BLOCK_ROW) * triton.cdiv(inner, BLOCK_COL),)
    _geglu_kernel[grid](x, y, num_rows, inner, BLOCK_ROW=BLOCK_ROW, BLOCK_COL=BLOCK_COL, num_warps=2, num_stages=3)
    return y
