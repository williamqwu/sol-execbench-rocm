import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def _run(
    input_features: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    embed_positions_weight: torch.Tensor,
    self_attn_layer_norm_weight: torch.Tensor,
    self_attn_layer_norm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    final_layer_norm_weight: torch.Tensor,
    final_layer_norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    d_model = 5120
    num_heads = 20
    head_dim = 256
    scaling = head_dim ** -0.5

    # conv1 via im2col + matmul (faster than conv1d on MI350X)
    B_a, C_in_a, L_a = input_features.shape
    K_a = conv1_weight.shape[2]; C_out_a = conv1_weight.shape[0]
    xpa = F.pad(input_features, (1, 1))
    colsa = xpa.unfold(2, K_a, 1).permute(0, 1, 3, 2).contiguous().view(B_a, C_in_a * K_a, -1)
    w1 = conv1_weight.view(C_out_a, C_in_a * K_a)
    x = torch.matmul(w1, colsa) + conv1_bias.view(1, -1, 1)
    x = F.gelu(x)
    # conv2 via im2col + matmul
    B_c, C_in_c, L_c = x.shape
    K_c = conv2_weight.shape[2]
    C_out_c = conv2_weight.shape[0]
    xp = F.pad(x, (1, 1))
    cols = xp.unfold(2, K_c, 2).permute(0, 1, 3, 2).contiguous().view(B_c, C_in_c * K_c, -1)
    w2 = conv2_weight.view(C_out_c, C_in_c * K_c)
    x = torch.matmul(w2, cols) + conv2_bias.view(1, -1, 1)
    x = F.gelu(x)
    x = x.permute(0, 2, 1)

    bsz, seq_len, _ = x.shape

    hidden_states = x + embed_positions_weight

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states.to(torch.float32),
        (d_model,),
        self_attn_layer_norm_weight.to(torch.float32),
        self_attn_layer_norm_bias.to(torch.float32),
        eps=1e-5,
    ).to(torch.bfloat16)

    # Fused QKV projection: stack weights -> one GEMM
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv_bias = torch.cat([q_proj_bias, torch.zeros_like(q_proj_bias), v_proj_bias], dim=0)
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    query_states, key_states, value_states = qkv.split(d_model, dim=-1)
    query_states = query_states * scaling

    q = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = value_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(q, k, v)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, d_model)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states.to(torch.float32),
        (d_model,),
        final_layer_norm_weight.to(torch.float32),
        final_layer_norm_bias.to(torch.float32),
        eps=1e-5,
    ).to(torch.bfloat16)

    hidden_states = F.linear(hidden_states, fc1_weight, fc1_bias)
    hidden_states = F.gelu(hidden_states)
    hidden_states = F.linear(hidden_states, fc2_weight, fc2_bias)
    hidden_states = residual + hidden_states

    return hidden_states


_run_compiled = torch.compile(_run, mode="reduce-overhead")


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    embed_positions_weight: torch.Tensor,
    self_attn_layer_norm_weight: torch.Tensor,
    self_attn_layer_norm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    final_layer_norm_weight: torch.Tensor,
    final_layer_norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    return _run_compiled(
        input_features,
        conv1_weight, conv1_bias,
        conv2_weight, conv2_bias,
        embed_positions_weight,
        self_attn_layer_norm_weight, self_attn_layer_norm_bias,
        q_proj_weight, q_proj_bias,
        k_proj_weight, v_proj_weight, v_proj_bias,
        out_proj_weight, out_proj_bias,
        final_layer_norm_weight, final_layer_norm_bias,
        fc1_weight, fc1_bias,
        fc2_weight, fc2_bias,
    )
