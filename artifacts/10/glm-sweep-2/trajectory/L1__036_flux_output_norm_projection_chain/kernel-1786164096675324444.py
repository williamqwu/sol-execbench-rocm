import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    inner_dim = temb.shape[-1]
    B, S, D = hidden_states.shape

    # LayerNorm without affine (keep exact math: 1/sqrt)
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states_norm = (hidden_states - mean) / torch.sqrt(var + eps)

    # SiLU on temb then modulation GEMM
    temb_silu = temb * torch.sigmoid(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    shift = modulation[:, :inner_dim].unsqueeze(1)
    scale = modulation[:, inner_dim:].unsqueeze(1)

    # Modulation
    hidden_states_mod = hidden_states_norm * (1.0 + scale) + shift

    # Output projection
    output = F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
    return output
