import torch
import triton
import triton.language as tl


@triton.jit
def _epilogue_kernel(gate_ptr, up_ptr, out_ptr, n_elements,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    ag = 0.5 * g * (2.0 * tl.sigmoid(2.0 * inner))
    out = (ag * u).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    gate_output = torch.matmul(x, gate_proj.t())
    up_output = torch.matmul(x, up_proj.t())
    out = torch.empty_like(gate_output)
    n = gate_output.numel()
    grid = (triton.cdiv(n, 4096),)
    _epilogue_kernel[grid](gate_output, up_output, out, n, BLOCK=4096)
    return out
