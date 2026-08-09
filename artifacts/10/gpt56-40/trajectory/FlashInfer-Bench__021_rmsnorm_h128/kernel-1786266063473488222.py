import torch
import torch._dynamo.config

torch._dynamo.config.recompile_limit = 32
torch._dynamo.config.cache_size_limit = 32


@torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
def _compiled_rmsnorm(hidden_states, weight):
    x = hidden_states.float()
    inv_rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1.0e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    return _compiled_rmsnorm(hidden_states, weight)
