import torch


@torch.compile
def _rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 2048
    return _rmsnorm(hidden_states, weight)
