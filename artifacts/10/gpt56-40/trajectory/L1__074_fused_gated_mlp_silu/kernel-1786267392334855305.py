import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gate_kernel(x, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // 4096
    col = offsets - row * 4096
    gate_off = row * 8192 + col
    up_off = gate_off + 4096
    gate = tl.load(x + gate_off, mask=mask)
    up = tl.load(x + up_off, mask=mask)
    value = up * gate * tl.sigmoid(gate)
    tl.store(x + up_off, value, mask=mask)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    n_elements = hidden_states.numel() * 4
    _gate_kernel[(triton.cdiv(n_elements, 256),)](gate_up, n_elements=n_elements, BLOCK=256)
    up = gate_up[..., 4096:]
    return F.linear(up, down_weight)
