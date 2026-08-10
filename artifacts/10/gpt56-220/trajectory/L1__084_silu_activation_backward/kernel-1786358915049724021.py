import torch
import triton
import triton.language as tl


@triton.jit
def _silu_bw(grad_output, x, sigmoid_x, out, n: tl.constexpr,
             BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    g = tl.load(grad_output + offsets, mask=mask)
    xv = tl.load(x + offsets, mask=mask)
    s = tl.load(sigmoid_x + offsets, mask=mask)
    y = g * s * (1.0 + xv * (1.0 - s))
    tl.store(out + offsets, y, mask=mask)


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(grad_output)
    n = grad_output.numel()
    _silu_bw[(triton.cdiv(n, 128),)](
        grad_output, x, sigmoid_x, out, n=n, BLOCK=128,
        num_warps=2,
    )
    return out
