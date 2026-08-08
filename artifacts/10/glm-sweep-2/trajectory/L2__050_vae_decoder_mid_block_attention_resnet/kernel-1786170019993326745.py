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
    head_dim = channels
    scale = head_dim ** -0.5
    seq_len = height * width

    # Keep feature maps in channels_last so MIOpen convs run natively in NHWC.
    if hidden_states.stride() != (channels * height * width, 1, width, channels):
        hidden_states = hidden_states.to(memory_format=torch.channels_last)
    else:
        hidden_states = hidden_states.to(memory_format=torch.channels_last)

    # Convert conv weights to channels_last once.
    r1c1w = resnet1_conv1_weight.to(memory_format=torch.channels_last)
    r1c2w = resnet1_conv2_weight.to(memory_format=torch.channels_last)
    r2c1w = resnet2_conv1_weight.to(memory_format=torch.channels_last)
    r2c2w = resnet2_conv2_weight.to(memory_format=torch.channels_last)

    def resnet_block(x, residual, n1w, n1b, c1w, c1b, tw, tb, n2w, n2b, c2w, c2b):
        h = F.group_norm(x, num_groups, n1w, n1b, eps)
        h = F.silu(h)
        h = F.conv2d(h, c1w, c1b, padding=1)
        temb_proj = F.silu(temb)
        temb_proj = F.linear(temb_proj, tw, tb)
        h = h + temb_proj[:, :, None, None]
        h = F.group_norm(h, num_groups, n2w, n2b, eps)
        h = F.silu(h)
        h = F.conv2d(h, c2w, c2b, padding=1)
        return h + residual

    # ============ ResNet Block 1 ============
    residual1 = hidden_states
    hidden_states = resnet_block(
        hidden_states, residual1,
        resnet1_norm1_weight, resnet1_norm1_bias,
        r1c1w, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
        resnet1_norm2_weight, resnet1_norm2_bias,
        r1c2w, resnet1_conv2_bias)

    # ============ Attention Block ============
    attn_residual = hidden_states
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    # h is channels_last [B,C,H,W]; reshape to [B, S, C] for attention
    h = h.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1).reshape(batch, seq_len, channels).contiguous()

    query = F.linear(h, attn_to_q_weight, attn_to_q_bias)
    key = F.linear(h, attn_to_k_weight, attn_to_k_bias)
    value = F.linear(h, attn_to_v_weight, attn_to_v_bias)

    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    h = torch.matmul(attention_probs, value)

    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)
    # reshape back to [B,C,H,W] channels_last
    h = h.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

    hidden_states = h + attn_residual

    # ============ ResNet Block 2 ============
    residual2 = hidden_states
    output = resnet_block(
        hidden_states, residual2,
        resnet2_norm1_weight, resnet2_norm1_bias,
        r2c1w, resnet2_conv1_bias,
        resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
        resnet2_norm2_weight, resnet2_norm2_bias,
        r2c2w, resnet2_conv2_bias)

    return output
