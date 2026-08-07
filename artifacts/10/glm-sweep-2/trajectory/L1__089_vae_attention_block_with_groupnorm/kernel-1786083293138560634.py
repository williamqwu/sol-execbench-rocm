import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = x.shape
    num_groups = 32
    seq_len = height * width

    # Group normalization
    channels_per_group = channels // num_groups
    x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_norm = (x_grouped - mean) * torch.rsqrt(var + eps)
    x_norm = x_norm.view(batch, channels, height, width)
    x_norm = x_norm * group_norm_weight.view(1, channels, 1, 1) + group_norm_bias.view(1, channels, 1, 1)

    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    x_seq = x_norm.permute(0, 2, 3, 1).reshape(batch, seq_len, channels).contiguous()

    # Q, K, V projections
    q = torch.matmul(x_seq, query_weight.t()) + query_bias
    k = torch.matmul(x_seq, key_weight.t()) + key_bias
    v = torch.matmul(x_seq, value_weight.t()) + value_bias

    # Fused attention via SDPA (avoids materializing [B, seq, seq] matrix)
    scale = channels ** -0.5
    attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)

    # Output projection
    attn_output = torch.matmul(attn_output, proj_out_weight.t()) + proj_out_bias

    # Reshape back to spatial format
    attn_output = attn_output.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

    return x + attn_output
