import torch
import torch.nn.functional as F
import torch._dynamo

# Keep group_norm as an exact eager op (preserves reduction order / numerics)
# while letting inductor fuse the surrounding elementwise ops (silu, adds, etc).
torch._dynamo.allow_in_graph(F.group_norm)


def _forward(
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
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1
    head_dim = channels
    scale = head_dim ** -0.5
    seq_len = height * width

    residual1 = hidden_states
    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    hidden_states = h + residual1

    attn_residual = hidden_states
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    h = h.view(batch, channels, seq_len).transpose(1, 2)
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
    h = h.transpose(1, 2).view(batch, channels, height, width)
    hidden_states = h + attn_residual

    residual2 = hidden_states
    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
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
