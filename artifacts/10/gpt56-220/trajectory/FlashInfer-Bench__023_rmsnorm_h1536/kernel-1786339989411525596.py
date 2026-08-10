import torch


@torch.compile
@torch.no_grad()
def run(hidden_states, weight):
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return ((x * inv_rms) * weight.to(torch.float32)).to(hidden_states.dtype)
