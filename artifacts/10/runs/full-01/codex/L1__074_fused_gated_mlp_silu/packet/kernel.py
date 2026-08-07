import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice as hip_libdevice


@triton.jit
def _double_gated_silu_kernel(gate_up, activated,
                              n_elements: tl.constexpr,
                              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // 4096
    col = offsets - row * 4096
    gate = tl.load(gate_up + row * 8192 + col, mask=mask)
    up = tl.load(gate_up + row * 8192 + 4096 + col, mask=mask)
    exponential = hip_libdevice.exp((-gate).to(tl.float64)).to(tl.float32)
    sigmoid_gate = 1.0 / (1.0 + exponential)
    silu_gate = gate * sigmoid_gate
    tl.store(activated + offsets, up * silu_gate, mask=mask)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    rows = hidden_states.numel() // 1024
    gate_up = torch.mm(hidden_states.view(rows, 1024), gate_up_weight.t())
    activated = torch.empty(
        (rows, 4096), device=hidden_states.device, dtype=hidden_states.dtype
    )
    n_elements = rows * 4096
    if rows == 256:
        _double_gated_silu_kernel[(triton.cdiv(n_elements, 512),)](
            gate_up, activated, n_elements=n_elements,
            BLOCK=512, num_warps=4,
        )
    else:
        _double_gated_silu_kernel[(triton.cdiv(n_elements, 256),)](
            gate_up, activated, n_elements=n_elements,
            BLOCK=256, num_warps=1,
        )
    output = gate_up.view(-1)[: rows * 1024].view(rows, 1024)
    torch.mm(activated, down_weight.t(), out=output)
    return output.view(*hidden_states.shape[:-1], 1024)
