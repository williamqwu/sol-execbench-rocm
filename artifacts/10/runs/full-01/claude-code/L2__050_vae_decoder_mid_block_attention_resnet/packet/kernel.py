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
    seq_len = height * width
    scale = channels ** -0.5

    # SiLU(temb) is identical in both ResNet blocks -- compute once.
    temb_act = F.silu(temb)

    # ============ ResNet Block 1 ============
    residual1 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)

    h = h + F.linear(temb_act, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)

    hidden_states = h + residual1

    # ============ Attention Block ============
    attn_residual = hidden_states

    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    h = h.view(batch, channels, seq_len).transpose(1, 2).contiguous()
    h2d = h.view(batch * seq_len, channels)

    # Single fused QKV GEMM instead of three separate ones. Bit-exact: each
    # output row is the same dot product, just batched into one launch.
    qkv_w = torch.cat((attn_to_q_weight, attn_to_k_weight, attn_to_v_weight), 0)
    qkv_b = torch.cat((attn_to_q_bias, attn_to_k_bias, attn_to_v_bias), 0)
    qkv = torch.addmm(qkv_b, h2d, qkv_w.t()).view(batch, seq_len, 3 * channels)
    query, key, value = qkv.split(channels, dim=-1)

    # num_heads == 1, so the [B, 1, S, D] reshape is a no-op and bmm suffices.
    scores = torch.bmm(query, key.transpose(-2, -1)).mul_(scale)
    torch.softmax(scores, dim=-1, out=scores)
    h = torch.bmm(scores, value)

    h = torch.addmm(attn_to_out_bias, h.reshape(batch * seq_len, channels),
                    attn_to_out_weight.t()).view(batch, seq_len, channels)
    h = h.transpose(1, 2).view(batch, channels, height, width)

    hidden_states = h + attn_residual

    # ============ ResNet Block 2 ============
    residual2 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)

    h = h + F.linear(temb_act, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)

    output = h + residual2

    return output
