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
    
    x_norm = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias, eps)
    
    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    
    # Compute Q, K, V projections using linear layers
    # q = x_seq @ query_weight.T + query_bias
    q = F.linear(x_seq, query_weight, query_bias)
    k = F.linear(x_seq, key_weight, key_bias)
    v = F.linear(x_seq, value_weight, value_bias)
    
    # Compute attention scores: Q @ K^T with scaling
    scale = channels ** -0.5
    attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, H*W, H*W)
    
    # Softmax over key dimension
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values
    attn_output = torch.bmm(attn_weights, v)  # (B, H*W, C)
    
    # Output projection
    attn_output = F.linear(attn_output, proj_out_weight, proj_out_bias)
    
    # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
    attn_output = attn_output.permute(0, 2, 1).contiguous()
    attn_output = attn_output.view(batch, channels, height, width)
    
    # Residual connection
    output = residual + attn_output
    
    return output
