import torch


@torch.compile(fullgraph=True, dynamic=True)
def _rmsnorm(hidden_states, weight):
    return torch.ops.aten.rms_norm.default(hidden_states, [4096], weight, 1e-5)


@torch.no_grad()
def run(hidden_states, weight):
    return _rmsnorm(hidden_states, weight)
