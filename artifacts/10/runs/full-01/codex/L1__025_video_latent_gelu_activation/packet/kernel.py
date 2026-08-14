import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gelu_cache_kernel(x_ptr, out_ptr, n_elements: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr, LOAD_CACHE: tl.constexpr,
                       STORE_CACHE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, cache_modifier=LOAD_CACHE,
                eviction_policy="evict_first")
    x2 = x * x
    x3 = x2 * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    out = (0.5 * x) * (1.0 + libdevice.tanh(inner))
    tl.store(out_ptr + offsets, out, mask=mask, cache_modifier=STORE_CACHE)


def run(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    _gelu_cache_kernel[(triton.cdiv(n_elements, 512),)](
        x, out, n_elements, BLOCK_SIZE=512, LOAD_CACHE=".cg",
        STORE_CACHE=".wt", num_warps=1
    )
    return out
