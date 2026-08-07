import torch
import triton
import triton.language as tl

@triton.jit
def _cos_sin_kernel(freqs_ptr, cos_ptr, sin_ptr, scaling,
                    n_elements, half_dim,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # freqs is [B, S, half_dim] flattened; one element -> two outputs
    f = tl.load(freqs_ptr + offs, mask=mask)
    c = tl.cos(f) * scaling
    s = tl.sin(f) * scaling
    # write into first half of output [.., head_dim]
    tl.store(cos_ptr + offs, c.to(tl.bfloat16), mask=mask)
    tl.store(sin_ptr + offs, s.to(tl.bfloat16), mask=mask)
    # write duplicated into second half (offset by n_elements)
    tl.store(cos_ptr + n_elements + offs, c.to(tl.bfloat16), mask=mask)
    tl.store(sin_ptr + n_elements + offs, s.to(tl.bfloat16), mask=mask)

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    B, S, H = freqs.shape
    n = B * S * H
    out_shape = (B, S, H * 2)
    cos = torch.empty(out_shape, device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty(out_shape, device=freqs.device, dtype=torch.bfloat16)
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _cos_sin_kernel[grid](freqs, cos, sin, attention_scaling, n, H, BLOCK=BLOCK)
    return cos, sin
