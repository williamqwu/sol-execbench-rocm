import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(grad_cos, grad_sin, theta, out, n: tl.constexpr,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(theta + offsets, mask=mask)
    gc = tl.load(grad_cos + offsets, mask=mask).to(tl.float32)
    gs = tl.load(grad_sin + offsets, mask=mask).to(tl.float32)
    value = -gc * tl.sin(x) + gs * tl.cos(x)
    tl.store(out + offsets, value, mask=mask)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(idx_theta)
    n = idx_theta.numel()
    _rope_kernel[(triton.cdiv(n, 256),)](
        grad_cos, grad_sin, idx_theta, out, n=n, BLOCK=256,
    )
    return out
