import torch


@torch.compile(fullgraph=True)
def _compiled_rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    mean_sq = torch.sum(x * x, dim=-1, keepdim=True) * (1.0 / 128.0)
    inv_rms = torch.rsqrt(mean_sq + 1.0e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    return _compiled_rmsnorm(hidden_states, weight)
