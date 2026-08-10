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
    
    # Self-attention block with gated residual
    residual = hidden_state
    
    # Input layer norm
    hidden_state = F.layer_norm(
        hidden_state,
        (hidden_size,),
        input_layernorm_weight,
        input_layernorm_bias,
        norm_eps,
    )
    
    # Compute Q, K, V
    query = F.linear(hidden_state, q_proj_weight)
    key = F.linear(hidden_state, k_proj_weight)
    value = F.linear(hidden_state, v_proj_weight)
    
    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Fused scaled dot-product attention avoids materializing the score matrix.
    attn_output = F.scaled_dot_product_attention(
        query, key, value, scale=scaling
    )
    
    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, o_proj_weight)
    
    # Gated residual connection for attention
    hidden_state = residual + torch.tanh(gate_attn) * attn_output
    
    # MLP block with gated residual
    residual = hidden_state
    
    # Post-attention layer norm
    hidden_state = F.layer_norm(
        hidden_state,
        (hidden_size,),
        post_attention_layernorm_weight,
        post_attention_layernorm_bias,
        norm_eps,
    )
    
    # MLP with GELU activation
    hidden_state = F.linear(hidden_state, fc1_weight, fc1_bias)
    hidden_state = F.gelu(hidden_state, approximate="tanh")
    hidden_state = F.linear(hidden_state, fc2_weight, fc2_bias)
    
    # Gated residual connection for MLP
    output = residual + torch.tanh(gate_ffn) * hidden_state
    
    return output
