import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(hidden_states, residual, weight):
    x = hidden_states.float() + residual.float()
    inv_rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1.0e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, residual, weight):
    assert hidden_states.shape[1] == 2048
    return _compiled(hidden_states, residual, weight)
