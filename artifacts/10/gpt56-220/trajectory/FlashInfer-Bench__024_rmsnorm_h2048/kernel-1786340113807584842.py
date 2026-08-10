import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 2048
    return F.rms_norm(hidden_states, (2048,), weight, eps=1e-6)
