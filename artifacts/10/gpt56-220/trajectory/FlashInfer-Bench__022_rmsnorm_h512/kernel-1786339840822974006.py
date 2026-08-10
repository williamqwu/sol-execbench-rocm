import torch

@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 512
    return torch.nn.functional.rms_norm(
        hidden_states, (512,), weight=weight, eps=1.0e-6
    )
