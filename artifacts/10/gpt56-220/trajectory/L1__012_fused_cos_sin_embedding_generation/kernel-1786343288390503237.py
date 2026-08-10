import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x, output, n_elements: tl.constexpr,
                 scale: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    value = tl.load(x + idx, mask=mask)
    c = tl.cos(value) * scale
    s = tl.sin(value) * scale
    row = idx // 64
    col = idx - row * 64
    out = row * 128 + col
    plane_size = n_elements * 2
    tl.store(output + out, c, mask=mask)
    tl.store(output + out + 64, c, mask=mask)
    tl.store(output + plane_size + out, s, mask=mask)
    tl.store(output + plane_size + out + 64, s, mask=mask)


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    n = freqs.numel()
    output_shape = (*freqs.shape[:-1], 128)
    output = torch.empty((2, *output_shape), device=freqs.device,
                         dtype=torch.bfloat16)
    _rope_kernel[(triton.cdiv(n, 512),)](
        freqs, output,
        n_elements=n,
        scale=attention_scaling,
        BLOCK=512,
        num_warps=1,
    )
    return output[0], output[1]
