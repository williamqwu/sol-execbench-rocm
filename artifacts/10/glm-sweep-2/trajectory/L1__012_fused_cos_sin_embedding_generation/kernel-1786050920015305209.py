import torch
import triton
import triton.language as tl

@triton.jit
def _cos_sin_kernel(freqs_ptr, cos_ptr, sin_ptr, scaling,
                    n_freq, HALF: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_freq
    # freq index -> output row/col
    row = offs // HALF
    col = offs % HALF
    base = row * (2 * HALF) + col
    f = tl.load(freqs_ptr + offs, mask=mask)
    c = (tl.cos(f) * scaling).to(tl.bfloat16)
    s = (tl.sin(f) * scaling).to(tl.bfloat16)
    tl.store(cos_ptr + base, c, mask=mask)
    tl.store(sin_ptr + base, s, mask=mask)
    tl.store(cos_ptr + base + HALF, c, mask=mask)
    tl.store(sin_ptr + base + HALF, s, mask=mask)

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    B, S, H = freqs.shape
    n_freq = B * S * H
    cos = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    BLOCK = 256
    grid = (triton.cdiv(n_freq, BLOCK),)
    _cos_sin_kernel[grid](freqs, cos, sin, attention_scaling, n_freq, HALF=H, BLOCK=BLOCK)
    return cos, sin
