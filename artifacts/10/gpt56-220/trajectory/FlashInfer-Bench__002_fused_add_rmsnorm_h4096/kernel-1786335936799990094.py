import torch


def _impl(hidden_states, residual, weight):
    x = hidden_states.float() + residual.float()
    inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1.0e-5)
    return (x * inv * weight.float()).to(hidden_states.dtype)


_compiled = torch.compile(_impl, fullgraph=True, dynamic=True)


@torch.no_grad()
def run(hidden_states, residual, weight):
    assert hidden_states.shape[1] == 4096
    return _compiled(hidden_states, residual, weight)
