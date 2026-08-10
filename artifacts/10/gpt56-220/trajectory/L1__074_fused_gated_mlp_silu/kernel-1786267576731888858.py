import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mul_gate_inplace(gate_up, sigmoid_gate, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // 4096
    col = offsets - row * 4096
    gate_offset = row * 8192 + col
    up_offset = gate_offset + 4096
    gate = tl.load(gate_up + gate_offset, mask=mask)
    sigmoid = tl.load(sigmoid_gate + offsets, mask=mask)
    up = tl.load(gate_up + up_offset, mask=mask)
    silu = gate * sigmoid
    result = up * silu
    tl.store(gate_up + up_offset, result, mask=mask)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid_gate = torch.sigmoid(gate)
    n_elements = hidden_states.numel() * 4
    _mul_gate_inplace[(triton.cdiv(n_elements, 256),)](
        gate_up, sigmoid_gate, n_elements=n_elements, BLOCK=256
    )
    return F.linear(up, down_weight)
