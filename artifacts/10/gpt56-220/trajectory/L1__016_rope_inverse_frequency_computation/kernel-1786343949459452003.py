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

_COMPILED = _rope_kernel.warmup(
    _OUTPUT_TEMPLATE, 1.0, grid=(1,), num_warps=1
)


@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty_like(_OUTPUT_TEMPLATE)
    _COMPILED.run(
        1, 1, 1, torch.cuda.current_stream().cuda_stream,
        _COMPILED.function, out, float(rope_theta)
    )
    return out
