import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    attn1_to_q_weight: torch.Tensor,
    attn1_to_k_weight: torch.Tensor,
    attn1_to_v_weight: torch.Tensor,
    attn1_to_out_weight: torch.Tensor,
    attn1_to_out_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    attn2_to_q_weight: torch.Tensor,
    attn2_to_k_weight: torch.Tensor,
    attn2_to_v_weight: torch.Tensor,
    attn2_to_out_weight: torch.Tensor,
    attn2_to_out_bias: torch.Tensor,
    norm3_weight: torch.Tensor,
    norm3_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    norm_eps: float,
):
    # Constants
    num_attention_heads = 160
    attention_head_dim = 24
    inner_dim = num_attention_heads * attention_head_dim
    scale = attention_head_dim ** -0.5
    
    batch_size = hidden_states.shape[0]
    spatial_seq_len = hidden_states.shape[1]
    encoder_seq_len = encoder_hidden_states.shape[1]
    
    # ============ Self-Attention Block ============
    # LayerNorm1
    mean1 = hidden_states.mean(dim=-1, keepdim=True)
    centered1 = hidden_states - mean1
    var1 = (centered1 ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = centered1 / torch.sqrt(var1 + norm_eps)
    norm_hidden_states = torch.addcmul(norm1_bias, norm_hidden_states, norm1_weight)
    
    # QKV projections for self-attention
    query = F.linear(norm_hidden_states, attn1_to_q_weight)
    key = F.linear(norm_hidden_states, attn1_to_k_weight)
    value = F.linear(norm_hidden_states, attn1_to_v_weight)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query = query.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    query = query * scale
    
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, attn1_to_out_weight, attn1_to_out_bias)
    
    # Residual connection
    hidden_states = hidden_states + attn_output
    
    # ============ Cross-Attention Block ============
    # LayerNorm2
    mean2 = hidden_states.mean(dim=-1, keepdim=True)
    centered2 = hidden_states - mean2
    var2 = (centered2 ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = centered2 / torch.sqrt(var2 + norm_eps)
    norm_hidden_states = torch.addcmul(norm2_bias, norm_hidden_states, norm2_weight)
    
    # QKV projections for cross-attention
    query = F.linear(norm_hidden_states, attn2_to_q_weight)
    key = F.linear(encoder_hidden_states, attn2_to_k_weight)
    value = F.linear(encoder_hidden_states, attn2_to_v_weight)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query = query.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    query = query * scale
    
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, attn2_to_out_weight, attn2_to_out_bias)
    
    # Residual connection
    hidden_states = hidden_states + attn_output
    
    # ============ Feed-Forward Block ============
    # LayerNorm3
    mean3 = hidden_states.mean(dim=-1, keepdim=True)
    centered3 = hidden_states - mean3
    var3 = (centered3 ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = centered3 / torch.sqrt(var3 + norm_eps)
    norm_hidden_states = torch.addcmul(norm3_bias, norm_hidden_states, norm3_weight)
    
    # First linear (produces 2x intermediate for GEGLU)
    ff_output = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    
    # GEGLU activation: split and apply gelu to gate
    x, gate = ff_output.chunk(2, dim=-1)
    ff_output = x * F.gelu(gate, approximate='tanh')
    
    # Second linear
    ff_output = F.linear(ff_output, ff_linear2_weight, ff_linear2_bias)
    
    # Residual connection
    output = hidden_states + ff_output
    
    return output
