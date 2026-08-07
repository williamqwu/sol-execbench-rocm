import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, n,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x_gate = tl.load(x_ptr + offs, mask=mask)
    x_lin = tl.load(x_ptr + offs + n, mask=mask)
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715
    inner = c * (x_gate + a * x_gate * x_gate * x_gate)
    gelu = 0.5 * x_gate * (1.0 + libdevice.tanh(inner))
    out = gelu * x_lin
    tl.store(out_ptr + offs, out, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    B, S, D2 = x.shape
    D = D2 // 2
    x_flat = x.reshape(-1, D2)
    rows = x_flat.shape[0]
    n = rows * D
    out = torch.empty((rows, D), device=x.device, dtype=x.dtype)
    x_c = x_flat.contiguous()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _geglu_kernel[grid](x_c, out, n, BLOCK=BLOCK)
    return out.view(B, S, D)
