import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_state: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    input_layernorm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    post_attention_layernorm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    gate_attn: torch.Tensor,
    gate_ffn: torch.Tensor,
    norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_state.shape
    num_heads = 16
    head_dim = hidden_size // num_heads
    scaling = head_dim ** -0.5

    residual = hidden_state

    hidden_state = F.layer_norm(
        hidden_state, (hidden_size,), input_layernorm_weight, input_layernorm_bias, norm_eps
    )

    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = F.linear(hidden_state, qkv_weight)
    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(query, key, value, scale=scaling)

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, o_proj_weight.t())

    gate_a = torch.tanh(gate_attn)
    hidden_state = torch.addcmul(residual, attn_output, gate_a)

    residual = hidden_state

    hidden_state = F.layer_norm(
        hidden_state, (hidden_size,), post_attention_layernorm_weight, post_attention_layernorm_bias, norm_eps
    )

    hidden_state = torch.matmul(hidden_state, fc1_weight.t()) + fc1_bias
    hidden_state = F.gelu(hidden_state)
    hidden_state = torch.matmul(hidden_state, fc2_weight.t()) + fc2_bias

    gate_f = torch.tanh(gate_ffn)
    output = torch.addcmul(residual, hidden_state, gate_f)

    return output
