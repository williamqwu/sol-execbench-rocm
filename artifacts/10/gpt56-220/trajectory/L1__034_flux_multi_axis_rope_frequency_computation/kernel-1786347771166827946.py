import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(ids, out_cos, out_sin, theta: tl.constexpr, n_elements: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    row = offs // 64
    col = offs - row * 64
    axis = tl.where(col < 8, 0, tl.where(col < 36, 1, 2))
    local = tl.where(col < 8, col, tl.where(col < 36, col - 8, col - 36))
    half = tl.where(axis == 0, 8.0, 28.0)
    exponent = local.to(tl.float32) / half
    pos = tl.load(ids + row * 3 + axis, mask=mask).to(tl.float32)
    angle = pos * tl.exp2(-exponent * tl.log2(theta))
    out_offs = row * 128 + col * 2
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    tl.store(out_cos + out_offs, cos, mask=mask)
    tl.store(out_cos + out_offs + 1, cos, mask=mask)
    tl.store(out_sin + out_offs, sin, mask=mask)
    tl.store(out_sin + out_offs + 1, sin, mask=mask)


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    n = ids.shape[0]
    out_cos = torch.empty((n, 128), device=ids.device, dtype=torch.float32)
    out_sin = torch.empty_like(out_cos)
    total = n * 64
    _rope_kernel[(triton.cdiv(total, 256),)](ids, out_cos, out_sin, theta=theta,
                                             n_elements=total, BLOCK=256)
    return out_cos, out_sin
