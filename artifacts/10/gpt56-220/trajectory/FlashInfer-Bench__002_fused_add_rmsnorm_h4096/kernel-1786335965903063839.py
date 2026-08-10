import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, residual, weight):
    assert hidden_states.shape[1] == 4096
    x = hidden_states.float() + residual.float()
    return F.rms_norm(x, (4096,), weight.float(), 1.0e-5).to(hidden_states.dtype)
