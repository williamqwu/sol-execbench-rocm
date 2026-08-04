import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inv_freq_kernel(out, rope_theta):
    offsets = tl.arange(0, 64)
    exponents = offsets.to(tl.float32) * 0.015625
    theta = rope_theta + 0.0
    values = tl.exp2(-tl.log2(theta) * exponents)
    tl.store(out + offsets, values)


@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty((64,), device="cuda", dtype=torch.float32)
    _rope_inv_freq_kernel[(1,)](out, float(rope_theta), num_warps=2)
    return out
