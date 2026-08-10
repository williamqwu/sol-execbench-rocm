import torch


@torch.no_grad()
def run(hidden_states, residual, weight):
    assert hidden_states.shape[1] == 2048
    x = hidden_states.float() + residual.float()
    return torch.nn.functional.rms_norm(
        x, (2048,), weight.float(), eps=1.0e-6
    ).to(hidden_states.dtype)
