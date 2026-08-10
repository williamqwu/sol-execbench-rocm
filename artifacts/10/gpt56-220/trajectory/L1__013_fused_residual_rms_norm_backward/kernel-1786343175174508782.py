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


@triton.jit
def _dw_kernel(go, norm, out, n_rows, H: tl.constexpr,
               BR: tl.constexpr, BC: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    rr = rb * BR + tl.arange(0, BR)[:, None]
    cc = cb * BC + tl.arange(0, BC)[None, :]
    mask = (rr < n_rows) & (cc < H)
    off = rr * H + cc
    g = tl.load(go + off, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(norm + off, mask=mask, other=0.0)
    part = tl.sum(g * n, axis=0)
    tl.atomic_add(out + cb * BC + tl.arange(0, BC), part,
                  mask=(cb * BC + tl.arange(0, BC)) < H)


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor,
        rstd: torch.Tensor, weight: torch.Tensor):
    rows = grad_output.numel() // 2560
    out0 = torch.empty_like(grad_output)
    out1 = torch.empty_like(grad_output)
    _dx_kernel[(rows,)](grad_output, normalized, rstd, weight, out0, out1,
                        rows, H=2560, BLOCK=4096, num_warps=8)
    grad_weight = torch.zeros_like(weight)
    _dw_kernel[(triton.cdiv(rows, 64), triton.cdiv(2560, 256))](
        grad_output, normalized, grad_weight, rows, H=2560, BR=64, BC=256,
        num_warps=8)
    return out0, out1, grad_weight
