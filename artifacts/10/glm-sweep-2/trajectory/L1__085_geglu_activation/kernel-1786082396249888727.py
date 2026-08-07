import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, D, D2,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    col_offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = col_offs < D
    row_base = row * D2
    x_gate = tl.load(x_ptr + row_base + col_offs, mask=mask)
    x_lin = tl.load(x_ptr + row_base + D + col_offs, mask=mask)
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715
    inner = c * (x_gate + a * x_gate * x_gate * x_gate)
    gelu = 0.5 * x_gate * (1.0 + libdevice.tanh(inner))
    out = gelu * x_lin
    tl.store(out_ptr + row * D + col_offs, out, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    B, S, D2 = x.shape
    D = D2 // 2
    x_flat = x.reshape(-1, D2)
    rows = x_flat.shape[0]
    out = torch.empty((rows, D), device=x.device, dtype=x.dtype)
    BLOCK = 1024
    grid = (rows, triton.cdiv(D, BLOCK))
    _geglu_kernel[grid](x_flat, out, D, D2, BLOCK=BLOCK)
    return out.view(B, S, D)
