import torch
import torch.nn.functional as F
import torch._dynamo

torch._dynamo.allow_in_graph(F.group_norm)


def _resnet_block(x, temb_act, n1w, n1b, c1w, c1b, tw, tb, n2w, n2b, c2w, c2b,
                  num_groups, eps, residual):
    h = F.group_norm(x, num_groups, n1w, n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, c1w, c1b, padding=1)
    temb_proj = F.linear(temb_act, tw, tb)
    h = h + temb_proj[:, :, None, None]
    h = F.group_norm(h, num_groups, n2w, n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, c2w, c2b, padding=1)
    return h + residual


_compiled_resnet = torch.compile(_resnet_block, dynamic=False)


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
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1
    head_dim = channels
    scale = head_dim ** -0.5
    seq_len = height * width

    temb_act = F.silu(temb)

    # ResNet Block 1 (compiled, group_norm kept eager)
    residual1 = hidden_states
    hidden_states = _compiled_resnet(
        hidden_states, temb_act,
        resnet1_norm1_weight, resnet1_norm1_bias,
        resnet1_conv1_weight, resnet1_conv1_bias,
        resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias,
        resnet1_norm2_weight, resnet1_norm2_bias,
        resnet1_conv2_weight, resnet1_conv2_bias,
        num_groups, eps, residual1)

    # Attention Block (fully eager — preserves exact numerics)
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

    # ResNet Block 2 (compiled)
    residual2 = hidden_states
    output = _compiled_resnet(
        hidden_states, temb_act,
        resnet2_norm1_weight, resnet2_norm1_bias,
        resnet2_conv1_weight, resnet2_conv1_bias,
        resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias,
        resnet2_norm2_weight, resnet2_norm2_bias,
        resnet2_conv2_weight, resnet2_conv2_bias,
        num_groups, eps, residual2)

    return output
