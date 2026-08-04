import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gate, up, out, n_elements:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + offsets, mask=mask, other=0.0)

    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    tanh_inner = 2.0 / (1.0 + tl.exp2(-2.8853900817779268 * inner)) - 1.0
    activated = (0.5 * g * (1.0 + tanh_inner)).to(tl.bfloat16)
    y = activated * u
    tl.store(out + offsets, y, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    gate_output = torch.matmul(x, gate_proj.t())
    up_output = torch.matmul(x, up_proj.t())
    n_elements = gate_output.numel()
    _swiglu_kernel[(triton.cdiv(n_elements, 2048),)](
        gate_output, up_output, gate_output, n_elements, BLOCK_SIZE=2048, num_warps=4
    )
    return gate_output
