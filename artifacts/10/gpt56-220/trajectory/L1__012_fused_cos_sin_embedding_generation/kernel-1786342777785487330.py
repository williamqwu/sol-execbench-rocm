import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x, cos_out, sin_out, n_elements: tl.constexpr,
                 scale: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    value = tl.load(x + idx, mask=mask)
    c = tl.cos(value) * scale
    s = tl.sin(value) * scale
    row = idx // 64
    col = idx - row * 64
    out = row * 128 + col
    tl.store(cos_out + out, c, mask=mask)
    tl.store(cos_out + out + 64, c, mask=mask)
    tl.store(sin_out + out, s, mask=mask)
    tl.store(sin_out + out + 64, s, mask=mask)


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    n = freqs.numel()
    shape = (*freqs.shape[:-1], 128)
    cos = torch.empty(shape, device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty_like(cos)
    _rope_kernel[(triton.cdiv(n, 256),)](
        freqs, cos, sin, n_elements=n, scale=attention_scaling, BLOCK=256,
        num_warps=4,
    )
    return cos, sin
