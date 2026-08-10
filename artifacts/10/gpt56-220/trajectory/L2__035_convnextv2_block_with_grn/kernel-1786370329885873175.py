import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    residual = x
    B, C, H, W = x.shape
    
    # Depthwise convolution: (B, C, H, W) -> (B, C, H, W)
    # groups=C for depthwise
    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=C)
    
    # Permute to channels_last: (B, C, H, W) -> (B, H, W, C)
    out = out.permute(0, 2, 3, 1)
    
    # LayerNorm: (B, H, W, C) -> (B, H, W, C)
    out = F.layer_norm(out, (C,), layernorm_weight, layernorm_bias, eps=layer_norm_eps)
    
    # Pointwise expansion: (B, H, W, C) -> (B, H, W, 4C)
    # Using matmul: out @ pwconv1_weight.T + pwconv1_bias
    out = F.linear(out, pwconv1_weight, pwconv1_bias)
    
    # GELU activation: (B, H, W, 4C) -> (B, H, W, 4C)
    out = torch.ops.aten.gelu_.default(out)
    
    # Global Response Normalization (GRN)
    # Compute L2 norm across spatial dimensions (H, W)
    # Shape: (B, H, W, 4C) -> (B, 1, 1, 4C)
    global_features = torch.norm(out, p=2, dim=(1, 2), keepdim=True)
    
    # Normalize by channel-wise mean
    # Shape: (B, 1, 1, 4C) -> (B, 1, 1, 4C)
    channel_mean = global_features.mean(dim=-1, keepdim=True)
    channel_mean.add_(eps)
    global_features.div_(channel_mean)
    norm_features = global_features
    
    # Apply GRN transformation with learnable parameters
    # x_grn = weight * (x * norm_features) + bias + x
    grn_out = out * norm_features
    grn_out.mul_(grn_weight)
    grn_out.add_(grn_bias)
    grn_out.add_(out)
    out = grn_out
    
    # Pointwise projection: (B, H, W, 4C) -> (B, H, W, C)
    out = F.linear(out, pwconv2_weight, pwconv2_bias)
    
    # Permute back to channels_first: (B, H, W, C) -> (B, C, H, W)
    out = out.permute(0, 3, 1, 2)
    
    # Residual connection (no drop path in inference)
    out.add_(residual)
    
    return out
