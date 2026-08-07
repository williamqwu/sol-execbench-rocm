import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

_D = 5120
_D2 = 10240
_BLOCK = 512


@triton.jit
def _geglu_kernel(x_ptr, out_ptr,
                  D: tl.constexpr, D2: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cb = tl.program_id(1)
    col_offs = cb * BLOCK + tl.arange(0, BLOCK)
    row_base = row * D2
    x_gate = tl.load(x_ptr + row_base + col_offs)
    x_lin = tl.load(x_ptr + row_base + D + col_offs)
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715
    inner = c * (x_gate + a * x_gate * x_gate * x_gate)
    gelu = 0.5 * x_gate * (1.0 + libdevice.tanh(inner))
    tl.store(out_ptr + row * D + col_offs, gelu * x_lin)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    B, S, D2 = x.shape
    D = D2 // 2
    x_flat = x.reshape(-1, D2)
    rows = x_flat.shape[0]
    out = torch.empty((rows, D), device=x.device, dtype=x.dtype)
    grid = (rows, D // _BLOCK)
    _geglu_kernel[grid](x_flat, out, D=D, D2=D2, BLOCK=_BLOCK, num_warps=4)
    return out.view(B, S, D)
