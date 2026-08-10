import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _geglu_fast(x, out, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = row * 10240 + col
    gate = tl.load(x + base)
    linear = tl.load(x + base + 5120)
    z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    e = tl.exp2(-2.8853900817779268 * tl.abs(z))
    mag = (1.0 - e) / (1.0 + e)
    tanh_z = tl.where(z >= 0.0, mag, -mag)
    value = 0.5 * gate * (1.0 + tanh_z) * linear
    tl.store(out + row * 5120 + col, value)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device,
                      dtype=x.dtype)
    rows = x.numel() // 10240
    _geglu_fast[(5, rows)](x, out, BLOCK=1024)
    return out
