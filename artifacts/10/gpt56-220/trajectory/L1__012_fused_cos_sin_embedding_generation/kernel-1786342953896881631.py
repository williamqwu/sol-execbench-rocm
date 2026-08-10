import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x, cos_out, sin_out, scale: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 64)
    value = tl.load(x + row * 64 + col)
    c = tl.cos(value) * scale
    s = tl.sin(value) * scale
    out = row * 128 + col
    tl.store(cos_out + out, c)
    tl.store(cos_out + out + 64, c)
    tl.store(sin_out + out, s)
    tl.store(sin_out + out + 64, s)


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    n_rows = freqs.numel() // 64
    output_shape = (*freqs.shape[:-1], 128)
    cos = torch.empty(output_shape, device=freqs.device, dtype=torch.bfloat16)
    sin = torch.empty_like(cos)
    _rope_kernel[(n_rows,)](
        freqs, cos, sin,
        scale=attention_scaling,
        num_warps=1,
    )
    return cos, sin
