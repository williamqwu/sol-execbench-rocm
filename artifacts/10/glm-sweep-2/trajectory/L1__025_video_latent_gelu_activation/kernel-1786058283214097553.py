import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from triton.language.extra.libdevice import tanh


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr, NPG: tl.constexpr):
    pid = tl.program_id(0)
    for j in tl.static_range(NPG):
        idx = pid * NPG + j
        offs = idx * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1.0 + tanh(inner))
        tl.store(y_ptr + offs, y, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if 25_000_000 <= n <= 50_000_000:
        x = x.contiguous()
        y = torch.empty_like(x)
        BLOCK, nw, ns, npg = 1024, 4, 3, 2
        grid = (triton.cdiv(n, BLOCK * npg),)
        _gelu_tanh_kernel[grid](x, y, n, BLOCK=BLOCK, num_warps=nw,
                                num_stages=ns, NPG=npg)
        return y
    return F.gelu(x, approximate='tanh')
