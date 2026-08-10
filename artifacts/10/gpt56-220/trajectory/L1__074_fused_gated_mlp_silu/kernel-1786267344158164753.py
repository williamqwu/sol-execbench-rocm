import torch
import torch.nn.functional as F


def _gate(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return up * (gate * torch.sigmoid(gate))


_compiled_gate = torch.compile(_gate, dynamic=True)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, gate_up_weight)
    activated = _compiled_gate(gate_up)
    return F.linear(activated, down_weight)
