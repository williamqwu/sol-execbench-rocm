import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(out, theta: tl.float32):
    i = tl.arange(0, 64)
    value = tl.exp((-i.to(tl.float32) * (1.0 / 64.0)) * tl.log(theta))
    tl.store(out + i, value)


@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty(64, dtype=torch.float32, device="cuda")
    _rope_kernel[(1,)](out, float(rope_theta), num_warps=1)
    return out
