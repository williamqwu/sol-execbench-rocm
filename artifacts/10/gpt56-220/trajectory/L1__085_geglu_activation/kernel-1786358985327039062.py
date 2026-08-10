import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _geglu_fast(x, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    row = offsets // 5120
    col = offsets - row * 5120
    base = row * 10240 + col
    gate = tl.load(x + base, mask=mask)
    linear = tl.load(x + base + 5120, mask=mask)
    z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    value = 0.5 * gate * (1.0 + libdevice.fast_tanhf(z)) * linear
    tl.store(out + offsets, value, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device,
                      dtype=x.dtype)
    n = out.numel()
    _geglu_fast[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
