import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, n, D, D2,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs // D
    col = offs % D
    base = row * D2
    x_gate = tl.load(x_ptr + base + col, mask=mask)
    x_lin = tl.load(x_ptr + base + D + col, mask=mask)
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715
    inner = c * (x_gate + a * x_gate * x_gate * x_gate)
    gelu = 0.5 * x_gate * (1.0 + libdevice.fast_tanhf(inner))
    tl.store(out_ptr + offs, gelu * x_lin, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    B, S, D2 = x.shape
    D = D2 // 2
    x_flat = x.reshape(-1, D2)
    rows = x_flat.shape[0]
    n = rows * D
    out = torch.empty((rows, D), device=x.device, dtype=x.dtype)
    # Adaptive config tuned on MI350X. Large workloads are memory-bandwidth
    # bound and prefer more warps for occupancy; smaller ones are launch-bound.
    if n > 20_000_000:
        BLOCK, num_warps = 1024, 8
    else:
        BLOCK, num_warps = 1024, 4
    grid = (triton.cdiv(n, BLOCK),)
    _geglu_kernel[grid](x_flat, out, n, D, D2, BLOCK=BLOCK, num_warps=num_warps)
    return out.view(B, S, D)
