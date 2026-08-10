import torch

@torch.compile(fullgraph=True, dynamic=True)
def _compiled(hidden_states, residual, weight):
    x = hidden_states.float() + residual.float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)

@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 7168

    return _compiled(hidden_states, residual, weight)
