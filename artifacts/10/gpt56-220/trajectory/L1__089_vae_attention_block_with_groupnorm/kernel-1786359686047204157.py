import torch
import torch.nn.functional as F

@torch.no_grad()
@torch.compile(fullgraph=True, dynamic=True)
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
    x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.view(batch, channels, height, width)
    x_norm = x_norm * group_norm_weight.view(1, channels, 1, 1) + group_norm_bias.view(1, channels, 1, 1)
    
    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    
    # Compute Q, K, V projections using linear layers
    # q = x_seq @ query_weight.T + query_bias
    q = F.linear(x_seq, query_weight, query_bias)
    k = F.linear(x_seq, key_weight, key_bias)
    v = F.linear(x_seq, value_weight, value_bias)
    
    scale = channels ** -0.5
    attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.bmm(attn_weights, v)
    
    # Output projection
    attn_output = F.linear(attn_output, proj_out_weight, proj_out_bias)
    
    # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
    attn_output = attn_output.permute(0, 2, 1).contiguous()
    attn_output = attn_output.view(batch, channels, height, width)
    
    # Residual connection
    output = residual + attn_output
    
    return output
