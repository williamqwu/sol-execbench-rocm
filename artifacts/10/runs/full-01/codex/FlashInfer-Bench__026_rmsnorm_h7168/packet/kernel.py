import torch
import aiter


@torch.no_grad()
def run(hidden_states, weight):
    return aiter.rms_norm(hidden_states, weight, 1.0e-6)
