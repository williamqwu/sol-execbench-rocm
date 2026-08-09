import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mul_gate_inplace(gate, up, sigmoid_gate, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate_value = tl.load(gate + offsets, mask=mask)
    sigmoid = tl.load(sigmoid_gate + offsets, mask=mask)
    up_value = tl.load(up + offsets, mask=mask)
    result = up_value * (gate_value * sigmoid)
    tl.store(up + offsets, result, mask=mask)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate = F.linear(hidden_states, gate_up_weight[:4096])
    up = F.linear(hidden_states, gate_up_weight[4096:])
    sigmoid_gate = torch.sigmoid(gate)
    n_elements = hidden_states.numel() * 4
    _mul_gate_inplace[(triton.cdiv(n_elements, 1024),)](
        gate, up, sigmoid_gate, n_elements=n_elements, BLOCK=1024
    )
    return F.linear(up, down_weight)
