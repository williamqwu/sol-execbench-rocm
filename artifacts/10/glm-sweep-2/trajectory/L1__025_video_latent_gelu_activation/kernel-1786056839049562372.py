import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    # Config chosen from per-shape sweep: best overall on MI350X wavefront=64.
    if n >= 67108864:        # >= 64M elements: large, bandwidth-bound
        BLOCK, nw, ns = 1024, 2, 3
    elif n >= 16777216:      # 16M-64M
        BLOCK, nw, ns = 2048, 4, 1
    else:                    # < 16M: launch-overhead bound, medium block
        BLOCK, nw, ns = 4096, 4, 1
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_tanh_kernel[grid](x, y, n, BLOCK=BLOCK, num_warps=nw, num_stages=ns)
    return y
