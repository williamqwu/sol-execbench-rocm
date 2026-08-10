import torch
import triton
import triton.language as tl


@triton.jit
def _pointwise(go, wb, act, out, n: tl.constexpr, lo, hi,
               BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n
    g = tl.load(go + offs, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(wb + offs, mask=valid, other=0.0).to(tl.float32)
    a = tl.load(act + offs, mask=valid, other=0.0).to(tl.float32)
    y = tl.where((a > lo) & (a < hi), g * tl.sigmoid(x), 0.0)
    tl.store(out + offs, y, mask=valid)


@torch.no_grad()
def run(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    n = grad_output.numel()
    out = torch.empty_like(grad_output)
    block = 256
    _pointwise[(triton.cdiv(n, block),)](
        grad_output, dt_with_bias, dt_activated, out, n,
        time_step_min, time_step_max, BLOCK=block,
    )
    return out, out.sum(dim=(0, 1))
