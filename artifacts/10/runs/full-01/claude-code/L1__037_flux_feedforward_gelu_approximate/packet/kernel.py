import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    x = F.linear(hidden_states, fc1_weight, fc1_bias)
    x = F.gelu(x, approximate="tanh")
    return F.linear(x, fc2_weight, fc2_bias)
