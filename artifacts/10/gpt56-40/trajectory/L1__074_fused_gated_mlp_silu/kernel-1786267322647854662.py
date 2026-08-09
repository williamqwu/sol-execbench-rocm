import torch
import torch.nn.functional as F


def _impl(hidden_states, gate_up_weight, down_weight):
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.linear(up * F.silu(gate), down_weight)


_compiled = torch.compile(_impl, dynamic=True)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    return _compiled(hidden_states, gate_up_weight, down_weight)
