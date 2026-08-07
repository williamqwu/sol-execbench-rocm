import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    x = F.linear(hidden_states, fc1_weight, fc1_bias)
    torch.ops.aten.gelu_.default(x, approximate="tanh")
    return F.linear(x, fc2_weight, fc2_bias)
