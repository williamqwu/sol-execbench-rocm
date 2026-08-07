import torch

@torch.compile(mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    mean = hidden_states.mean(dim=-1, keepdim=True)
    variance = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (hidden_states - mean) / torch.sqrt(variance + eps)
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    inner_dim = hidden_states.shape[-1]
    scale = modulation[:, :inner_dim].unsqueeze(1)
    shift = modulation[:, inner_dim:].unsqueeze(1)
    output = normalized * (1.0 + scale) + shift
    return output
