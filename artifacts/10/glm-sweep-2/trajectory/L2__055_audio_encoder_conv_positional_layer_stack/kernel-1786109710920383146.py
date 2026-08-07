import torch
import torch.nn.functional as F
import math


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
    # Constants
    d_model = 5120
    num_heads = 20
    head_dim = 256
    scaling = head_dim ** -0.5

    # Stage 1: Convolutional feature extraction
    x = F.conv1d(input_features, conv1_weight, conv1_bias, padding=1)
    x = F.gelu(x)
    x = F.conv1d(x, conv2_weight, conv2_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Permute to (B, 1500, 5120)
    x = x.permute(0, 2, 1)

    bsz, seq_len, _ = x.shape

    # Stage 2: Add positional embeddings
    hidden_states = x + embed_positions_weight

    # Stage 3: Self-attention block (pre-norm residual)
    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states.to(torch.float32),
        (d_model,),
        self_attn_layer_norm_weight.to(torch.float32),
        self_attn_layer_norm_bias.to(torch.float32),
        eps=1e-5,
    ).to(torch.bfloat16)

    # Q, K, V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias) * scaling
    key_states = F.linear(hidden_states, k_proj_weight, None)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    # Reshape for multi-head attention: (B, seq_len, d_model) -> (B, num_heads, seq_len, head_dim)
    q = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = value_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    # Flash attention (fused softmax + matmul)
    attn_output = F.scaled_dot_product_attention(q, k, v)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, d_model)

    # Output projection
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    hidden_states = residual + attn_output

    # Stage 4: FFN block
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
