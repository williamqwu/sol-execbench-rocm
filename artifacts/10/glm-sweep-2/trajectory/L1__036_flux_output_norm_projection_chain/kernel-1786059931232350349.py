import torch
import torch.nn.functional as F

_compiled = None

def _run(hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, eps):
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states_norm = (hidden_states - mean) / torch.sqrt(var + eps)
    temb_silu = temb * torch.sigmoid(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    inner_dim = temb.shape[-1]
    shift = modulation[:, :inner_dim].unsqueeze(1)
    scale = modulation[:, inner_dim:].unsqueeze(1)
    hidden_states_mod = hidden_states_norm * (1.0 + scale) + shift
    output = F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
    return output

_compiled = torch.compile(_run, dynamic=True, mode="default")

@torch.no_grad()
def run(hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, eps):
    return _compiled(hidden_states, temb, linear_weight, linear_bias, proj_out_weight, proj_out_bias, eps)
