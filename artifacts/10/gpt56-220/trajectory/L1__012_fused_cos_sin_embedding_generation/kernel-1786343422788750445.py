import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x, cos_out, sin_out, n_elements: tl.constexpr,
                 scale: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    idx = pid * BLOCK + local
    mask = idx < n_elements
    value = tl.load(x + idx, mask=mask)
    c = tl.cos(value) * scale
    s = tl.sin(value) * scale
    local_row = local // 64
    row = pid * (BLOCK // 64) + local_row
    col = local - local_row * 64
    out = row * 128 + col
    tl.store(cos_out + out, c, mask=mask)
    tl.store(cos_out + out + 64, c, mask=mask)
    tl.store(sin_out + out, s, mask=mask)
    tl.store(sin_out + out + 64, s, mask=mask)


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    n = freqs.numel()
    output_shape = (*freqs.shape[:-1], 128)
    cos = torch.empty(output_shape, device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty_like(cos)
    _rope_kernel[(triton.cdiv(n, 512),)](
        freqs, cos, sin,
        n_elements=n,
        scale=attention_scaling,
        BLOCK=512,
        num_warps=1,
    )
    return cos, sin
