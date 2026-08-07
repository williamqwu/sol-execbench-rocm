import torch
import torch.nn.functional as F

@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def _fwd(
    hidden_states: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
):
    x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)
    x = F.gelu(x)
    global_features = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)
    x = grn_weight * (x * norm_features) + grn_bias + x
    output = F.linear(x, pwconv2_weight, pwconv2_bias)
    return output

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
):
    return _fwd(hidden_states, pwconv1_weight, pwconv1_bias, grn_weight,
                grn_bias, pwconv2_weight, pwconv2_bias, eps)
