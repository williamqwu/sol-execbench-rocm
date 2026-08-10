import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(ids, out_cos, out_sin, theta: tl.constexpr, n_elements: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    row = offs // 128
    col = offs - row * 128
    axis = tl.where(col < 16, 0, tl.where(col < 72, 1, 2))
    local = tl.where(col < 16, col, tl.where(col < 72, col - 16, col - 72))
    half = tl.where(axis == 0, 8.0, 28.0)
    exponent = (local // 2).to(tl.float32) / half
    pos = tl.load(ids + row * 3 + axis, mask=mask).to(tl.float32)
    angle = pos * tl.exp2(-exponent * tl.log2(theta))
    tl.store(out_cos + offs, tl.cos(angle), mask=mask)
    tl.store(out_sin + offs, tl.sin(angle), mask=mask)


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    n = ids.shape[0]
    out = torch.empty((2, n, 128), device=ids.device, dtype=torch.float32)
    out_cos = out[0]
    out_sin = out[1]
    total = n * 128
    _rope_kernel[(triton.cdiv(total, 256),)](ids, out_cos, out_sin, theta=theta,
                                             n_elements=total, BLOCK=256)
    return out_cos, out_sin
