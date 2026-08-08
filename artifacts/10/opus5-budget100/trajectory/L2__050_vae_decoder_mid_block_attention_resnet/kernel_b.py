import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states, temb,
    resnet1_norm1_weight, resnet1_norm1_bias, resnet1_conv1_weight, resnet1_conv1_bias,
    resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
    resnet1_norm2_weight, resnet1_norm2_bias, resnet1_conv2_weight, resnet1_conv2_bias,
    attn_group_norm_weight, attn_group_norm_bias,
    attn_to_q_weight, attn_to_q_bias, attn_to_k_weight, attn_to_k_bias,
    attn_to_v_weight, attn_to_v_bias, attn_to_out_weight, attn_to_out_bias,
    resnet2_norm1_weight, resnet2_norm1_bias, resnet2_conv1_weight, resnet2_conv1_bias,
    resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
    resnet2_norm2_weight, resnet2_norm2_bias, resnet2_conv2_weight, resnet2_conv2_bias,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    scale = channels ** -0.5

    residual1 = hidden_states
    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)
    temb_proj = F.linear(F.silu(temb), resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    hidden_states = h + residual1

    attn_residual = hidden_states
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    seq_len = height * width
    # work in [B, C, S] layout: no transposes needed
    hb = h.view(batch, channels, seq_len)

    w_qkv = torch.cat((attn_to_q_weight, attn_to_k_weight, attn_to_v_weight), 0)
    b_qkv = torch.cat((attn_to_q_bias, attn_to_k_bias, attn_to_v_bias), 0)
    qkv = torch.matmul(w_qkv, hb) + b_qkv[None, :, None]      # [B, 3C, S]
    query, key, value = qkv.split(channels, dim=1)             # each [B, C, S]

    scores = torch.matmul(query.transpose(1, 2), key) * scale  # [B, S, S]
    probs = F.softmax(scores, dim=-1)
    ctx = torch.matmul(value, probs.transpose(1, 2))           # [B, C, S]
    h = (torch.matmul(attn_to_out_weight, ctx) + attn_to_out_bias[None, :, None]).view(
        batch, channels, height, width)
    hidden_states = h + attn_residual

    residual2 = hidden_states
    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)
    temb_proj = F.linear(F.silu(temb), resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
    return h + residual2
