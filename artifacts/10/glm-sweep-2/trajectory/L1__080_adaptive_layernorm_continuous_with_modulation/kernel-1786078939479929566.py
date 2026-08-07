import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    # Fused layernorm: mean/var/normalize in one kernel
    normalized = torch.nn.functional.layer_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)

    # temb projection -> scale and shift
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    inner_dim = hidden_states.shape[-1]
    scale = modulation[:, :inner_dim].unsqueeze(1)
    shift = modulation[:, inner_dim:].unsqueeze(1)

    # Fused modulation: normalized * (1 + scale) + shift
    output = normalized * scale + normalized + shift
    return output
