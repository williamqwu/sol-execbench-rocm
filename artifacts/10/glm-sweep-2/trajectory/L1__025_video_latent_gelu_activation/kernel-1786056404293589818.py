import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # GELU tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_tanh_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y
