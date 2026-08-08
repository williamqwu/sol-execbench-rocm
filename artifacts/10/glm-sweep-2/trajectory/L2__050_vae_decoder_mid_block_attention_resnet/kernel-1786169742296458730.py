import torch
import torch.nn.functional as F


def _forward(
    hidden_states, temb,
    r1n1w, r1n1b, r1c1w, r1c1b, r1tw, r1tb, r1n2w, r1n2b, r1c2w, r1c2b,
    agnw, agnb, aqw, aqb, akw, akb, avw, avb, aow, aob,
    r2n1w, r2n1b, r2c1w, r2c1b, r2tw, r2tb, r2n2w, r2n2b, r2c2w, r2c2b,
    eps,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1
    head_dim = channels
    scale = head_dim ** -0.5
    seq_len = height * width

    # ResNet Block 1
    residual1 = hidden_states
    h = F.group_norm(hidden_states, num_groups, r1n1w, r1n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c1w, r1c1b, padding=1)
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, r1tw, r1tb)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, r1n2w, r1n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c2w, r1c2b, padding=1)
    hidden_states = h + residual1

    # Attention Block
    attn_residual = hidden_states
    h = F.group_norm(hidden_states, num_groups, agnw, agnb, eps)
    h = h.view(batch, channels, seq_len).transpose(1, 2)
    query = F.linear(h, aqw, aqb)
    key = F.linear(h, akw, akb)
    value = F.linear(h, avw, avb)
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    h = torch.matmul(attention_probs, value)
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    h = F.linear(h, aow, aob)
    h = h.transpose(1, 2).view(batch, channels, height, width)
    hidden_states = h + attn_residual

    # ResNet Block 2
    residual2 = hidden_states
    h = F.group_norm(hidden_states, num_groups, r2n1w, r2n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c1w, r2c1b, padding=1)
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, r2tw, r2tb)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, r2n2w, r2n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c2w, r2c2b, padding=1)
    output = h + residual2
    return output


_compiled = torch.compile(_forward, dynamic=False)


@torch.no_grad()
def run(
    hidden_states, temb,
    resnet1_norm1_weight, resnet1_norm1_bias,
    resnet1_conv1_weight, resnet1_conv1_bias,
    resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
    resnet1_norm2_weight, resnet1_norm2_bias,
    resnet1_conv2_weight, resnet1_conv2_bias,
    attn_group_norm_weight, attn_group_norm_bias,
    attn_to_q_weight, attn_to_q_bias,
    attn_to_k_weight, attn_to_k_bias,
    attn_to_v_weight, attn_to_v_bias,
    attn_to_out_weight, attn_to_out_bias,
    resnet2_norm1_weight, resnet2_norm1_bias,
    resnet2_conv1_weight, resnet2_conv1_bias,
    resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
    resnet2_norm2_weight, resnet2_norm2_bias,
    resnet2_conv2_weight, resnet2_conv2_bias,
    eps,
):
    return _compiled(
        hidden_states, temb,
        resnet1_norm1_weight, resnet1_norm1_bias,
        resnet1_conv1_weight, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
        resnet1_norm2_weight, resnet1_norm2_bias,
        resnet1_conv2_weight, resnet1_conv2_bias,
        attn_group_norm_weight, attn_group_norm_bias,
        attn_to_q_weight, attn_to_q_bias,
        attn_to_k_weight, attn_to_k_bias,
        attn_to_v_weight, attn_to_v_bias,
        attn_to_out_weight, attn_to_out_bias,
        resnet2_norm1_weight, resnet2_norm1_bias,
        resnet2_conv1_weight, resnet2_conv1_bias,
        resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
        resnet2_norm2_weight, resnet2_norm2_bias,
        resnet2_conv2_weight, resnet2_conv2_bias,
        eps,
    )
