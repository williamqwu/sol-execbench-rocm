import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, weight):
    return F.rms_norm(hidden_states, (1536,), weight, 1e-6)
