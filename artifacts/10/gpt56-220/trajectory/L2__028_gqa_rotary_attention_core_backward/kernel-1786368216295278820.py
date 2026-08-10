import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for GQA with RoPE.
    
    Computes gradients through:
    1. Output projection
    2. Attention mechanism (softmax, matmul)
    3. GQA head expansion
    4. RoPE application
    5. QKV projections
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 8
    device = hidden_states.device
    original_dtype = hidden_states.dtype
    
    # ========== FORWARD RECOMPUTATION ==========
    # QKV projections
    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)
    
    # Reshape to multi-head format
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Compute RoPE embeddings
    position_ids = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    inv_freq_expanded = inv_freq[None, :, None].float()
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    # The reference forms [freqs, freqs], evaluates trig on all 128 values,
    # then retains only the first 64 before interleaving.  Avoid that dead half.
    cos_interleaved = freqs.cos().unsqueeze(1).repeat_interleave(2, dim=-1)
    sin_interleaved = freqs.sin().unsqueeze(1).repeat_interleave(2, dim=-1)
    
    # Apply rotary position embeddings
    q_float = query_states.float()
    k_float = key_states.float()
    
    # Rotate half for query and key
    q1 = q_float[..., 0::2]
    q2 = q_float[..., 1::2]
    k1 = k_float[..., 0::2]
    k2 = k_float[..., 1::2]

    q_rotated = torch.stack((-q2, q1), dim=-1).flatten(-2)
    k_rotated = torch.stack((-k2, k1), dim=-1).flatten(-2)

    query_states_rope = (q_float * cos_interleaved) + (q_rotated * sin_interleaved)
    key_states_rope = (k_float * cos_interleaved) + (k_rotated * sin_interleaved)
    
    query_states_rope = query_states_rope.to(original_dtype)
    key_states_rope = key_states_rope.to(original_dtype)
    
    # Expand key/value heads
    key_states_expanded = key_states_rope[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    value_states_expanded = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states_rope, key_states_expanded.transpose(2, 3))
    attn_weights.mul_(scaling)
    
    # Apply causal mask
    causal_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu_(diagonal=1)
    attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights_dtype = attn_weights_float.to(original_dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights_dtype, value_states_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # ========== BACKWARD PASS ==========
    # Gradient through output projection
    grad_attn_output = F.linear(grad_output, o_weight.t())
    grad_o_weight = grad_output.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, hidden_size)
    
    # Reshape gradient for attention output
    grad_attn_output = grad_attn_output.view(batch_size, seq_len, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # Gradient through attention matmul: attn_output = attn_weights @ value_states_expanded
    grad_attn_weights = torch.matmul(grad_attn_output, value_states_expanded.transpose(2, 3))
    grad_value_states_expanded = torch.matmul(attn_weights_dtype.transpose(2, 3), grad_attn_output)
    
    # Gradient through softmax
    grad_attn_weights_float = grad_attn_weights.float()
    attn_weights_float_saved = attn_weights_float
    grad_attn_scores = attn_weights_float_saved * (grad_attn_weights_float - (grad_attn_weights_float * attn_weights_float_saved).sum(dim=-1, keepdim=True))
    
    # Gradient through scaling
    grad_attn_scores.mul_(scaling)
    
    # Gradient through attention matmul: attn_scores = query_states_rope @ key_states_expanded.T
    grad_attn_scores_dtype = grad_attn_scores.to(original_dtype)
    grad_query_states_rope = torch.matmul(grad_attn_scores_dtype, key_states_expanded)
    grad_key_states_expanded = torch.matmul(query_states_rope.transpose(2, 3), grad_attn_scores_dtype).transpose(2, 3)
    
    # Gradient through key-value expansion
    grad_key_states_rope = grad_key_states_expanded.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    
    grad_value_states = grad_value_states_expanded.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    
    # Gradient through RoPE for query
    grad_query_states_rope_float = grad_query_states_rope.float()
    
    grad_q_float = grad_query_states_rope_float * cos_interleaved
    grad_q_rotated = grad_query_states_rope_float * sin_interleaved
    grad_q_rotated_reshaped = grad_q_rotated.view(*grad_q_rotated.shape[:-1], head_dim // 2, 2)
    grad_q1_from_rotated = grad_q_rotated_reshaped[..., 1]
    grad_q2_from_rotated = -grad_q_rotated_reshaped[..., 0]
    grad_q_float[..., 0::2] = grad_q_float[..., 0::2] + grad_q1_from_rotated
    grad_q_float[..., 1::2] = grad_q_float[..., 1::2] + grad_q2_from_rotated
    
    grad_query_states = grad_q_float.to(original_dtype)
    
    # Gradient through RoPE for key
    grad_key_states_rope_float = grad_key_states_rope.float()
    
    grad_k_float = grad_key_states_rope_float * cos_interleaved
    grad_k_rotated = grad_key_states_rope_float * sin_interleaved
    grad_k_rotated_reshaped = grad_k_rotated.view(*grad_k_rotated.shape[:-1], head_dim // 2, 2)
    grad_k1_from_rotated = grad_k_rotated_reshaped[..., 1]
    grad_k2_from_rotated = -grad_k_rotated_reshaped[..., 0]
    grad_k_float[..., 0::2] = grad_k_float[..., 0::2] + grad_k1_from_rotated
    grad_k_float[..., 1::2] = grad_k_float[..., 1::2] + grad_k2_from_rotated
    
    grad_key_states = grad_k_float.to(original_dtype)
    
    # Gradient through reshape and transpose for Q, K, V
    grad_query_states = grad_query_states.transpose(1, 2).contiguous()
    grad_key_states = grad_key_states.transpose(1, 2).contiguous()
    grad_value_states = grad_value_states.transpose(1, 2).contiguous()
    
    grad_query_states = grad_query_states.view(batch_size, seq_len, hidden_size)
    grad_key_states = grad_key_states.view(batch_size, seq_len, num_key_value_heads * head_dim)
    grad_value_states = grad_value_states.view(batch_size, seq_len, num_key_value_heads * head_dim)
    
    # Gradient through QKV projections
    grad_hidden_states_q = F.linear(grad_query_states, q_weight.t())
    grad_kv_states_for_input = torch.cat((grad_key_states, grad_value_states), dim=-1)
    kv_weight_for_input = torch.cat((k_weight, v_weight), dim=0)
    grad_hidden_states_kv = F.linear(grad_kv_states_for_input, kv_weight_for_input.t())

    grad_hidden_states_q.add_(grad_hidden_states_kv)
    grad_hidden_states = grad_hidden_states_q
    
    grad_q_weight = grad_query_states.reshape(-1, hidden_size).t() @ hidden_states.reshape(-1, hidden_size)
    grad_kv_states = torch.cat((grad_key_states, grad_value_states), dim=-1)
    grad_kv_weight = grad_kv_states.reshape(-1, 2 * num_key_value_heads * head_dim).t() @ hidden_states.reshape(-1, hidden_size)
    grad_k_weight, grad_v_weight = grad_kv_weight.split(num_key_value_heads * head_dim, dim=0)
    
    return (
        grad_hidden_states.to(original_dtype),
        grad_q_weight.to(original_dtype),
        grad_k_weight.to(original_dtype),
        grad_v_weight.to(original_dtype),
        grad_o_weight.to(original_dtype)
    )
