import torch
import triton
import triton.language as tl
import aiter
from triton.language.extra import libdevice


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements: tl.constexpr,
                   BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Both GEMM outputs have already rounded to bfloat16.  Keep the activation
    # in fp32, explicitly round it back to bf16, then perform the bf16 product.
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    activated = (0.5 * gate * (1.0 + libdevice.tanh(inner))).to(tl.bfloat16)
    result = activated * up
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    n = gate_proj.shape[0]

    # For larger matrices hipBLASLt can write into the two halves of a
    # row-interleaved buffer directly.  AITER then consumes that layout with a
    # single vectorized activation/multiply pass.
    if m >= 512:
        x_2d = x.view(m, x.shape[-1])
        projections = torch.empty((m, 2 * n), device=x.device, dtype=x.dtype)
        torch.mm(x_2d, gate_proj.t(), out=projections[:, :n])
        torch.mm(x_2d, up_proj.t(), out=projections[:, n:])
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        aiter.gelu_tanh_and_mul(out, projections)
        return out.view(*x.shape[:-1], n)

    gate = torch.matmul(x, gate_proj.t())
    up = torch.matmul(x, up_proj.t())
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(n_elements, 1024),)](
        gate, up, out, n_elements=n_elements, BLOCK=1024, num_warps=2
    )
    return out
