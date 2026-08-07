import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, D: tl.constexpr, D2: tl.constexpr,
                  BLOCK: tl.constexpr, ROWS: tl.constexpr):
    r0 = tl.program_id(0) * ROWS
    cb = tl.program_id(1)
    col_offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = col_offs < D
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715
    for r in tl.static_range(ROWS):
        row_base = (r0 + r) * D2
        x_gate = tl.load(x_ptr + row_base + col_offs, mask=mask)
        x_lin = tl.load(x_ptr + row_base + D + col_offs, mask=mask)
        inner = c * (x_gate + a * x_gate * x_gate * x_gate)
        gelu = 0.5 * x_gate * (1.0 + libdevice.fast_tanhf(inner))
        tl.store(out_ptr + (r0 + r) * D + col_offs, gelu * x_lin, mask=mask)


_D = 5120
_D2 = 10240


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    B, S, D2 = x.shape
    D = D2 // 2
    x_flat = x.reshape(-1, D2)
    rows = x_flat.shape[0]
    n = rows * D
    out = torch.empty((rows, D), device=x.device, dtype=x.dtype)
    if n > 20_000_000:
        BLOCK, num_warps, ROWS = 1024, 8, 1
    else:
        BLOCK, num_warps, ROWS = 1024, 4, 1
    grid = (triton.cdiv(rows, ROWS), triton.cdiv(D, BLOCK))
    _geglu_kernel[grid](x_flat, out, D=D, D2=D2, BLOCK=BLOCK, ROWS=ROWS,
                        num_warps=num_warps)
    return out.view(B, S, D)
