import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gate_inplace(gate_up, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate_offset = row * 8192 + col
    up_offset = gate_offset + 4096
    gate = tl.load(gate_up + gate_offset)
    up = tl.load(gate_up + up_offset)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate))
    silu = gate * sigmoid
    result = up * silu
    tl.store(gate_up + up_offset, result)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    n_elements = hidden_states.numel() * 4
    _gate_inplace[(n_elements // 4096, 4)](gate_up, BLOCK=1024)
    return F.linear(up, down_weight)
