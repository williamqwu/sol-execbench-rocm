import torch
import torch.nn.functional as F


def _mul_gate(gate, up, sigmoid_gate):
    return up * (gate * sigmoid_gate)


_compiled_mul_gate = torch.compile(_mul_gate, dynamic=False)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = _compiled_mul_gate(gate, up, torch.sigmoid(gate))
    return F.linear(activated, down_weight)
