import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.libdevice.tanh(inner))
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _gelu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
