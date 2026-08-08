import torch

_compiled = None

def _fn(hidden_states, weight):
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)

_compiled = torch.compile(_fn, mode="max-autotune", fullgraph=True, dynamic=False)


@torch.no_grad()
def run(hidden_states, weight):
    return _compiled(hidden_states, weight)
