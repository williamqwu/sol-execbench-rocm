import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _geglu_kernel(x, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // 5120
    col = offsets - row * 5120
    base = row * 10240 + col
    gate = tl.load(x + base, mask=mask)
    linear = tl.load(x + base + 5120, mask=mask)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + libdevice.tanh(inner))
    tl.store(out + offsets, gelu * linear, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device,
                      dtype=x.dtype)
    n = out.numel()
    _geglu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
