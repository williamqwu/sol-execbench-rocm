import torch


@torch.compile(dynamic=True, fullgraph=True)
def _compiled_rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    inv = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1.0e-6)
    return (x * inv * weight.float()).to(torch.bfloat16)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 7168
    return _compiled_rmsnorm(hidden_states, weight)
