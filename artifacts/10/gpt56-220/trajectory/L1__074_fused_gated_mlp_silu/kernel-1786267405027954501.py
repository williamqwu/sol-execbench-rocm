import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    gate.mul_(torch.sigmoid(gate))
    up.mul_(gate)
    return F.linear(up, down_weight)
