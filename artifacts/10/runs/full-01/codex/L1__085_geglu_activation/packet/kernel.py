import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


INNER_DIM = 5120
INPUT_DIM = 10240


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, INNER: tl.constexpr,
                  BLOCK: tl.constexpr, WRITE_THROUGH: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    gate = tl.load(x_ptr + row * (2 * INNER) + cols,
                   cache_modifier=".cg")
    linear = tl.load(x_ptr + row * (2 * INNER) + INNER + cols,
                     cache_modifier=".cg")

    gate3 = gate * gate * gate
    arg = 0.7978845608028654 * (gate + 0.044715 * gate3)
    gelu = 0.5 * gate * (1.0 + libdevice.tanh(arg))
    if WRITE_THROUGH:
        tl.store(out_ptr + row * INNER + cols, gelu * linear,
                 cache_modifier=".wt")
    else:
        tl.store(out_ptr + row * INNER + cols, gelu * linear,
                 cache_modifier=".cs")


def run(x: torch.Tensor) -> torch.Tensor:
    rows = x.numel() // INPUT_DIM
    out = torch.empty((*x.shape[:-1], INNER_DIM), device=x.device, dtype=x.dtype)
    if 256 < rows <= 512:
        _geglu_kernel[(10, rows)](
            x, out, INNER_DIM, BLOCK=512,
            WRITE_THROUGH=(rows == 512), num_warps=2
        )
    else:
        _geglu_kernel[(20, rows)](
            x, out, INNER_DIM, BLOCK=256,
            WRITE_THROUGH=(512 <= rows <= 2500), num_warps=1
        )
    return out
