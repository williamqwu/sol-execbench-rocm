import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _timestep_embedding_kernel(timestep, freqs, output, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < 384
    scaled = tl.load(timestep + row) * 1000.0
    arg = scaled * tl.load(freqs + col, mask=mask, other=0.0)
    tl.store(output + row * 768 + col, tl.cos(arg), mask=mask)
    tl.store(output + row * 768 + 384 + col, tl.sin(arg), mask=mask)


@triton.jit
def _silu_libdevice_kernel(x, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    value = tl.load(x + offset, mask=mask)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-value))
    tl.store(x + offset, value * sigmoid, mask=mask)


@torch.no_grad()
def run(
    timestep,
    pooled_projections,
    freqs,
    timestep_linear1_weight,
    timestep_linear1_bias,
    timestep_linear2_weight,
    timestep_linear2_bias,
    text_embedder_weight,
    text_embedder_bias,
):
    batch_size = timestep.shape[0]
    timestep_embed = torch.empty(
        (batch_size, 768), dtype=timestep.dtype, device=timestep.device
    )
    _timestep_embedding_kernel[(batch_size,)](
        timestep, freqs, timestep_embed, BLOCK=512, num_warps=1
    )
    x = torch.nn.functional.linear(
        timestep_embed, timestep_linear1_weight, timestep_linear1_bias
    )
    n_elements = x.numel()
    _silu_libdevice_kernel[(triton.cdiv(n_elements, 256),)](
        x, n_elements=n_elements, BLOCK=256, num_warps=4
    )
    timestep_embed = torch.nn.functional.linear(
        x, timestep_linear2_weight, timestep_linear2_bias
    )
    text_embed = torch.nn.functional.linear(
        pooled_projections, text_embedder_weight, text_embedder_bias
    )
    return timestep_embed + text_embed
