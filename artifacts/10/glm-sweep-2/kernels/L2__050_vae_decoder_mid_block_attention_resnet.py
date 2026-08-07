import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _group_norm_silu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C_channels, HW, cpg, eps,
    BLOCK_HW: tl.constexpr, BLOCK_CPG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    c_start = pid_g * cpg
    offs_c = c_start + tl.arange(0, BLOCK_CPG)
    offs_hw = tl.arange(0, BLOCK_HW)
    base = pid_b * C_channels * HW
    ptrs = X_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask = (offs_c[:, None] < C_channels) & (offs_hw[None, :] < HW)
    x = tl.load(ptrs, mask=mask, other=0.0)
    n_elem = cpg * HW
    mean = tl.sum(x) / n_elem
    d = x - mean
    var = tl.sum(d * d) / n_elem
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = d * rstd
    w = tl.load(W_ptr + offs_c, mask=offs_c < C_channels, other=0.0)
    b = tl.load(B_ptr + offs_c, mask=offs_c < C_channels, other=0.0)
    y = xn * w[:, None] + b[:, None]
    y = y * (1.0 / (1.0 + tl.exp(-y)))
    tl.store(Y_ptr + base + offs_c[:, None] * HW + offs_hw[None, :], y, mask=mask)


@triton.jit
def _group_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C_channels, HW, cpg, eps,
    BLOCK_HW: tl.constexpr, BLOCK_CPG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    c_start = pid_g * cpg
    offs_c = c_start + tl.arange(0, BLOCK_CPG)
    offs_hw = tl.arange(0, BLOCK_HW)
    base = pid_b * C_channels * HW
    ptrs = X_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask = (offs_c[:, None] < C_channels) & (offs_hw[None, :] < HW)
    x = tl.load(ptrs, mask=mask, other=0.0)
    n_elem = cpg * HW
    mean = tl.sum(x) / n_elem
    d = x - mean
    var = tl.sum(d * d) / n_elem
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = d * rstd
    w = tl.load(W_ptr + offs_c, mask=offs_c < C_channels, other=0.0)
    b = tl.load(B_ptr + offs_c, mask=offs_c < C_channels, other=0.0)
    y = xn * w[:, None] + b[:, None]
    tl.store(Y_ptr + base + offs_c[:, None] * HW + offs_hw[None, :], y, mask=mask)


def _group_norm_silu(x, w, b, num_groups, eps):
    B, C, H, W = x.shape
    HW = H * W
    y = torch.empty_like(x)
    cpg = C // num_groups
    BLOCK_HW = triton.next_power_of_2(HW)
    BLOCK_CPG = triton.next_power_of_2(cpg)
    _group_norm_silu_kernel[(B, num_groups)](x, w, b, y, C, HW, cpg, eps, BLOCK_HW=BLOCK_HW, BLOCK_CPG=BLOCK_CPG)
    return y


def _group_norm(x, w, b, num_groups, eps):
    B, C, H, W = x.shape
    HW = H * W
    y = torch.empty_like(x)
    cpg = C // num_groups
    BLOCK_HW = triton.next_power_of_2(HW)
    BLOCK_CPG = triton.next_power_of_2(cpg)
    _group_norm_kernel[(B, num_groups)](x, w, b, y, C, HW, cpg, eps, BLOCK_HW=BLOCK_HW, BLOCK_CPG=BLOCK_CPG)
    return y


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

    # ============ ResNet Block 1 ============
    residual1 = hidden_states
    h = _group_norm_silu(hidden_states, resnet1_norm1_weight, resnet1_norm1_bias, num_groups, eps)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)

    temb_proj = F.linear(F.silu(temb), resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]

    h = _group_norm_silu(h, resnet1_norm2_weight, resnet1_norm2_bias, num_groups, eps)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    hidden_states = h + residual1

    # ============ Attention Block ============
    attn_residual = hidden_states
    h = _group_norm(hidden_states, attn_group_norm_weight, attn_group_norm_bias, num_groups, eps)
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

    # ============ ResNet Block 2 ============
    residual2 = hidden_states
    h = _group_norm_silu(hidden_states, resnet2_norm1_weight, resnet2_norm1_bias, num_groups, eps)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)

    temb_proj = F.linear(F.silu(temb), resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]

    h = _group_norm_silu(h, resnet2_norm2_weight, resnet2_norm2_bias, num_groups, eps)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
    output = h + residual2

    return output
