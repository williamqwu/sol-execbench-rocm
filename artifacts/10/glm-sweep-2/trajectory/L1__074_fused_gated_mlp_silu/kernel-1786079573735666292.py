import torch
import torch.nn.functional as F


@torch.compile(fullgraph=False, dynamic=True)
def _ew(up):
    gate, up = up.chunk(2, dim=-1)
    silu_gate = gate * torch.sigmoid(gate)
    return up * silu_gate


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    return F.linear(_ew(F.linear(hidden_states, gate_up_weight)), down_weight)
