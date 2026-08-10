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
    
    # Store residual
    residual = x
    
    channels_per_group = channels // num_groups
    x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    var.add_(eps).sqrt_()
    x_norm = x_grouped - mean
    x_norm.div_(var)
    x_norm = x_norm.view(batch, channels, height, width)
    x_norm = torch.addcmul(
        group_norm_bias.view(1, channels, 1, 1),
        x_norm,
        group_norm_weight.view(1, channels, 1, 1),
    )
    
    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    
    # Compute Q, K, V projections using linear layers
    # q = x_seq @ query_weight.T + query_bias
    x_2d = x_seq.view(batch * seq_len, channels)
    q = torch.addmm(
        query_bias,
        x_2d,
        query_weight.t(),
        out=x_norm.view(batch * seq_len, channels),
    ).view(batch, seq_len, channels)
    k = torch.addmm(key_bias, x_2d, key_weight.t()).view(batch, seq_len, channels)
    v = torch.addmm(value_bias, x_2d, value_weight.t()).view(batch, seq_len, channels)
    
    scale = channels ** -0.5
    attn_scores = torch.baddbmm(
        query_bias[:1],
        q,
        k.transpose(1, 2),
        beta=0.0,
        alpha=scale,
    )
    attn_weights = torch.ops.aten._softmax.out(
        attn_scores, -1, False, out=attn_scores
    )
    attn_output = torch.bmm(attn_weights, v, out=q)
    
    # Output projection
    attn_output = torch.addmm(
        proj_out_bias,
        attn_output.view(batch * seq_len, channels),
        proj_out_weight.t(),
        out=k.view(batch * seq_len, channels),
    ).view(batch, seq_len, channels)
    
    # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
    attn_output = attn_output.permute(0, 2, 1)
    attn_output = attn_output.view(batch, channels, height, width)
    
    # Residual connection
    attn_output.add_(residual)
    output = attn_output
    
    return output
