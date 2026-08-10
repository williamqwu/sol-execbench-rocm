import torch
import torch._dynamo.config

torch._dynamo.config.recompile_limit = 32
torch._dynamo.config.cache_size_limit = 32


@torch.no_grad()
@torch.compile(fullgraph=True, dynamic=False)
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 128
    x = hidden_states.float()
    inv_rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1.0e-6)
    return (x * inv_rms * weight.float()).to(hidden_states.dtype)
