import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_OUTPUT_TEMPLATE = torch.empty(64, dtype=torch.float32, device="cuda")


@triton.jit
def _rope_kernel(out, theta: tl.float32):
    i = tl.arange(0, 64)
    exponent = -i.to(tl.float32) * (1.0 / 64.0)
    value = libdevice.pow(theta, exponent)
    tl.store(out + i, value)


def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty_like(_OUTPUT_TEMPLATE)
    _rope_kernel[(1,)](out, float(rope_theta), num_warps=1)
    return out
