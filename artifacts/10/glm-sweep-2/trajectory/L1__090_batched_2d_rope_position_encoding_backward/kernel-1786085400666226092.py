import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bwd_kernel(
    grad_cos_ptr, grad_sin_ptr, idx_theta_ptr, out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    idx_theta = tl.load(idx_theta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sin_t = tl.sin(idx_theta)
    cos_t = tl.cos(idx_theta)

    grad_cos = tl.load(grad_cos_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_sin = tl.load(grad_sin_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = -grad_cos * sin_t + grad_sin * cos_t
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(idx_theta)
    n = idx_theta.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    if n >= 5_000_000:
        _rope_bwd_kernel[grid](
            grad_cos, grad_sin, idx_theta, out, n,
            BLOCK=BLOCK, num_warps=4, num_stages=1,
        )
    else:
        _rope_bwd_kernel[grid](
            grad_cos, grad_sin, idx_theta, out, n,
            BLOCK=BLOCK, num_warps=8, num_stages=2,
        )
    return out
