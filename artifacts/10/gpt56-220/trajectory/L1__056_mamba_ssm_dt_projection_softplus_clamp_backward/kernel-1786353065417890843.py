import torch
import triton
import triton.language as tl


@triton.jit
def _fused(go, wb, act, out, bias, m: tl.constexpr, lo, hi,
           BLOCK: tl.constexpr):
    h = tl.program_id(0)
    p = tl.program_id(1)
    rows = p * BLOCK + tl.arange(0, BLOCK)
    valid = rows < m
    offs = rows * 40 + h
    g = tl.load(go + offs, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(wb + offs, mask=valid, other=0.0).to(tl.float32)
    a = tl.load(act + offs, mask=valid, other=0.0).to(tl.float32)
    y = tl.where((a > lo) & (a < hi), g * tl.sigmoid(x), 0.0)
    tl.store(out + offs, y, mask=valid)
    part = tl.sum(y, axis=0)
    tl.atomic_add(bias + h, part)


@torch.no_grad()
def run(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    m = grad_output.numel() // 40
    out = torch.empty_like(grad_output)
    bias32 = torch.zeros((40,), device=grad_output.device, dtype=torch.float32)
    block = 256
    _fused[(40, triton.cdiv(m, block))](
        grad_output, dt_with_bias, dt_activated, out, bias32, m,
        time_step_min, time_step_max, BLOCK=block,
    )
    return out, bias32.to(torch.bfloat16)
