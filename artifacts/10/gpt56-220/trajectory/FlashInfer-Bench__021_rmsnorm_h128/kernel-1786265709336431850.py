import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _compiled_rmsnorm(hidden_states, weight):
    return F.rms_norm(hidden_states, (128,), weight, 1.0e-6)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    return _compiled_rmsnorm(hidden_states, weight)
