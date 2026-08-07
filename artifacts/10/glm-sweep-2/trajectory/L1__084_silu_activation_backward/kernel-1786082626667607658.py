import torch
import triton
import triton.language as tl


@triton.jit
def _silu_bw_kernel(
    grad_out_ptr, x_ptr, sig_ptr, grad_in_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(grad_out_ptr + offs, mask=mask)
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(sig_ptr + offs, mask=mask)
    # grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))
    out = g * s * (1.0 + x * (1.0 - s))
    tl.store(grad_in_ptr + offs, out, mask=mask)


def _block_for(n):
    if n >= 1 << 22:      # >= 4M elements: bandwidth-bound, smaller blocks
        return 1024
    if n >= 1 << 19:      # >= 512K
        return 1024
    return 1024


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    grad_input = torch.empty_like(grad_output)
    n = grad_output.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _silu_bw_kernel[grid](
        grad_output, x, sigmoid_x, grad_input, n, BLOCK=BLOCK,
        num_warps=4, num_stages=3,
    )
    return grad_input
