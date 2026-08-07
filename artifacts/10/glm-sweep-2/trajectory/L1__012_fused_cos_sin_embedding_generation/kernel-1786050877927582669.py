import torch
import triton
import triton.language as tl

@triton.jit
def _cos_sin_kernel(freqs_ptr, cos_ptr, sin_ptr, scaling,
                    n_out, HALF: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    # each output element (in a row of width 2*HALF) maps to freqs[col % HALF]
    row = offs // (2 * HALF)
    col = offs % (2 * HALF)
    src = row * HALF + (col % HALF)
    f = tl.load(freqs_ptr + src, mask=mask)
    c = tl.cos(f) * scaling
    s = tl.sin(f) * scaling
    tl.store(cos_ptr + offs, c.to(tl.bfloat16), mask=mask)
    tl.store(sin_ptr + offs, s.to(tl.bfloat16), mask=mask)

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    B, S, H = freqs.shape
    n_out = B * S * (H * 2)
    cos = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _cos_sin_kernel[grid](freqs, cos, sin, attention_scaling, n_out, HALF=H, BLOCK=BLOCK)
    return cos, sin
