import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    return F.rms_norm(hidden_states, (128,), weight, 1.0e-6)
