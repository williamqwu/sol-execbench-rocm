import torch
import triton
import triton.language as tl


@triton.jit
def _dx_kernel(go, norm, rstd, weight, out0, out1, n_rows: tl.constexpr,
               H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = row * H + cols
    g = tl.load(go + off, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(norm + off, mask=mask, other=0.0)
    w = tl.load(weight + cols, mask=mask, other=0.0)
    gn = g * w
    mean = tl.sum(gn * n, axis=0) / H
    rs = tl.load(rstd + row)
    dx = rs * (gn - mean * n)
    tl.store(out0 + off, dx, mask=mask)
    tl.store(out1 + off, dx, mask=mask)


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor,
        rstd: torch.Tensor, weight: torch.Tensor):
    rows = grad_output.numel() // 2560
    out0 = torch.empty_like(grad_output)
    out1 = torch.empty_like(grad_output)
    _dx_kernel[(rows,)](grad_output, normalized, rstd, weight, out0, out1,
                        rows, H=2560, BLOCK=4096, num_warps=8)
    grad_weight = (grad_output.float() * normalized).sum(dim=(0, 1))
    return out0, out1, grad_weight
