import torch


@torch.compile(fullgraph=True)
def _compiled_rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    mean_sq = (x * x).mean(dim=-1, keepdim=True) + 1.0e-6
    inv_rms = torch.reciprocal(torch.sqrt(mean_sq))
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    return _compiled_rmsnorm(hidden_states, weight)
