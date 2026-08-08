import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 512
    return F.rms_norm(hidden_states, [H], weight, 1e-6)
