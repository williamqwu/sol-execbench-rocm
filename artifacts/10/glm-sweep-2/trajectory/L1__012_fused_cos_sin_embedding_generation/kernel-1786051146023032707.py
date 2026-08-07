import torch
import triton
import triton.language as tl

@triton.jit
def _cos_sin_kernel(freqs_ptr, cos_ptr, sin_ptr, scaling,
                    n_rows, HALF: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    mask = row < n_rows
    # load entire row of freqs: [HALF]
    f_offs = row[:, None] * HALF + tl.arange(0, HALF)[None, :]
    f = tl.load(freqs_ptr + f_offs, mask=mask[:, None])
    c = (tl.cos(f) * scaling).to(tl.bfloat16)
    s = (tl.sin(f) * scaling).to(tl.bfloat16)
    out_offs = row[:, None] * (2 * HALF) + tl.arange(0, HALF)[None, :]
    tl.store(cos_ptr + out_offs, c, mask=mask[:, None])
    tl.store(sin_ptr + out_offs, s, mask=mask[:, None])
    tl.store(cos_ptr + out_offs + HALF, c, mask=mask[:, None])
    tl.store(sin_ptr + out_offs + HALF, s, mask=mask[:, None])

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    B, S, H = freqs.shape
    n_rows = B * S
    cos = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty((B, S, H * 2), device=freqs.device, dtype=torch.bfloat16)
    BLOCK = 8
    grid = (triton.cdiv(n_rows, BLOCK),)
    _cos_sin_kernel[grid](freqs, cos, sin, attention_scaling, n_rows, HALF=H, BLOCK=BLOCK)
    return cos, sin
