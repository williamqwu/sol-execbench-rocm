import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096
    return F.rms_norm(hidden_states, [hidden_size], weight, 1e-5)
