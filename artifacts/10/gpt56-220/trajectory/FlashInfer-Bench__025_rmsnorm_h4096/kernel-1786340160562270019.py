import torch


@torch.compile(fullgraph=True, dynamic=True)
def _rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    return _rmsnorm(hidden_states, weight)
