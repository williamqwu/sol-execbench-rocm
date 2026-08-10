import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def _norm_grn(x, global_features, grn_weight, grn_bias, eps):
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)
    return grn_weight * (x * norm_features) + grn_bias + x

@torch.inference_mode()
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
    # Expansion linear: (B, H, W, dim) -> (B, H, W, hidden_dim)
    # F.linear computes x @ weight.T + bias
    x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)
    
    # GELU activation
    x = F.gelu(x, approximate="tanh")
    
    # Global Response Normalization (GRN)
    # Compute L2 norm across spatial dimensions (H, W)
    # Shape: (B, H, W, hidden_dim) -> (B, 1, 1, hidden_dim)
    global_features = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    # Normalize by channel-wise mean: (B, 1, 1, hidden_dim) -> (B, 1, 1, hidden_dim)
    # Apply learnable affine transformation with residual connection
    # weight * (input * norm_features) + bias + input
    x = _norm_grn(x, global_features, grn_weight, grn_bias, eps)
    
    # Projection linear: (B, H, W, hidden_dim) -> (B, H, W, dim)
    output = F.linear(x, pwconv2_weight, pwconv2_bias)

    return output
