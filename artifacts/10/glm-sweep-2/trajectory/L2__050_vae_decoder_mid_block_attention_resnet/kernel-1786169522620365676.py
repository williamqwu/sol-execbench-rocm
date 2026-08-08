import torch
import torch.nn.functional as F

@torch.no_grad()
def _resnet_block(x, temb, norm1_w, norm1_b, conv1_w, conv1_b, time_w, time_b,
                  norm2_w, norm2_b, conv2_w, conv2_b, num_groups, eps, residual):
    h = F.group_norm(x, num_groups, norm1_w, norm1_b, eps)
    h = F.silu(h)
    h = F.conv2d(h, conv1_w, conv1_b, padding=1)
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, time_w, time_b)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, norm2_w, norm2_b, eps)
    h = F.silu(h)
    h = F.conv2d(h, conv2_w, conv2_b, padding=1)
    return h + residual


def _attention(h, batch, channels, height, width, num_groups, eps,
               gn_w, gn_b, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
               num_heads, head_dim, scale, attn_residual):
    seq_len = height * width
    h = F.group_norm(h, num_groups, gn_w, gn_b, eps)
    h = h.view(batch, channels, seq_len).transpose(1, 2)
    query = F.linear(h, q_w, q_b)
    key = F.linear(h, k_w, k_b)
    value = F.linear(h, v_w, v_b)
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    h = torch.matmul(attention_probs, value)
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    h = F.linear(h, out_w, out_b)
    h = h.transpose(1, 2).view(batch, channels, height, width)
    return h + attn_residual


_compiled_resnet = torch.compile(_resnet_block, mode="max-autotune", dynamic=False)
_compiled_attn = torch.compile(_attention, mode="max-autotune", dynamic=False)


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

    residual1 = hidden_states
    hidden_states = _compiled_resnet(
        hidden_states, temb,
        resnet1_norm1_weight, resnet1_norm1_bias,
        resnet1_conv1_weight, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
        resnet1_norm2_weight, resnet1_norm2_bias,
        resnet1_conv2_weight, resnet1_conv2_bias,
        num_groups, eps, residual1)

    attn_residual = hidden_states
    hidden_states = _compiled_attn(
        hidden_states, batch, channels, height, width, num_groups, eps,
        attn_group_norm_weight, attn_group_norm_bias,
        attn_to_q_weight, attn_to_q_bias,
        attn_to_k_weight, attn_to_k_bias,
        attn_to_v_weight, attn_to_v_bias,
        attn_to_out_weight, attn_to_out_bias,
        num_heads, head_dim, scale, attn_residual)

    residual2 = hidden_states
    output = _compiled_resnet(
        hidden_states, temb,
        resnet2_norm1_weight, resnet2_norm1_bias,
        resnet2_conv1_weight, resnet2_conv1_bias,
        resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
        resnet2_norm2_weight, resnet2_norm2_bias,
        resnet2_conv2_weight, resnet2_conv2_bias,
        num_groups, eps, residual2)

    return output
