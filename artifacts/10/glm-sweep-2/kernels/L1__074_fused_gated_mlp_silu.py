import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    up_states = F.linear(hidden_states, gate_up_weight)
    gate, up_states = up_states.chunk(2, dim=-1)
    silu_gate = gate * torch.sigmoid(gate)
    up_states = up_states * silu_gate
    output = F.linear(up_states, down_weight)
    return output
