import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    resnet1_norm1_weight: torch.Tensor,
    resnet1_norm1_bias: torch.Tensor,
    resnet1_conv1_weight: torch.Tensor,
    resnet1_conv1_bias: torch.Tensor,
    resnet1_time_emb_proj_weight: torch.Tensor,
    resnet1_time_emb_proj_bias: torch.Tensor,
    resnet1_norm2_weight: torch.Tensor,
    resnet1_norm2_bias: torch.Tensor,
    resnet1_conv2_weight: torch.Tensor,
    resnet1_conv2_bias: torch.Tensor,
    attn_group_norm_weight: torch.Tensor,
    attn_group_norm_bias: torch.Tensor,
    attn_to_q_weight: torch.Tensor,
    attn_to_q_bias: torch.Tensor,
    attn_to_k_weight: torch.Tensor,
    attn_to_k_bias: torch.Tensor,
    attn_to_v_weight: torch.Tensor,
    attn_to_v_bias: torch.Tensor,
    attn_to_out_weight: torch.Tensor,
    attn_to_out_bias: torch.Tensor,
    resnet2_norm1_weight: torch.Tensor,
    resnet2_norm1_bias: torch.Tensor,
    resnet2_conv1_weight: torch.Tensor,
    resnet2_conv1_bias: torch.Tensor,
    resnet2_time_emb_proj_weight: torch.Tensor,
    resnet2_time_emb_proj_bias: torch.Tensor,
    resnet2_norm2_weight: torch.Tensor,
    resnet2_norm2_bias: torch.Tensor,
    resnet2_conv2_weight: torch.Tensor,
    resnet2_conv2_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1
    seq_len = height * width
    scale = channels ** -0.5

    # ============ ResNet Block 1 ============
    residual1 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)

    temb_act = F.silu(temb)

    temb_proj1 = F.linear(temb_act, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj1[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)

    hidden_states = h + residual1

    # ============ Attention Block (SDPA + fused QKV) ============
    attn_residual = hidden_states

    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)

    # [B, C, H, W] -> [B, H*W, C]
    h = h.reshape(batch, channels, seq_len).transpose(1, 2)

    # Fused QKV projection: concat weights along output dim
    qkv_w = torch.cat([attn_to_q_weight, attn_to_k_weight, attn_to_v_weight], dim=0)
    qkv_b = torch.cat([attn_to_q_bias, attn_to_k_bias, attn_to_v_bias], dim=0)
    qkv = F.linear(h, qkv_w, qkv_b)
    query, key, value = qkv.split(channels, dim=-1)

    # [B, H*W, C] -> [B, num_heads, H*W, head_dim]
    query = query.transpose(1, 2).view(batch, num_heads, seq_len, channels)
    key = key.transpose(1, 2).view(batch, num_heads, seq_len, channels)
    value = value.transpose(1, 2).view(batch, num_heads, seq_len, channels)

    h = F.scaled_dot_product_attention(query, key, value, scale=scale, dropout_p=0.0)

    # [B, num_heads, H*W, head_dim] -> [B, H*W, C]
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)

    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)

    # [B, H*W, C] -> [B, C, H, W]
    h = h.transpose(1, 2).reshape(batch, channels, height, width)

    hidden_states = h + attn_residual

    # ============ ResNet Block 2 ============
    residual2 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)

    temb_proj2 = F.linear(temb_act, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj2[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)

    output = h + residual2

    return output
