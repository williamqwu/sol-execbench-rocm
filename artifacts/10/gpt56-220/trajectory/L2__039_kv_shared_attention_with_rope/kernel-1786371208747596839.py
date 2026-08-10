import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    num_key_value_heads = axes_and_scalars["num_key_value_heads"]
    head_dim = axes_and_scalars["head_dim"]
    
    qkv_out_dim = num_attention_heads * head_dim
    kv_out_dim = num_key_value_heads * head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create causal mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = attention_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    q_proj_weight = torch.randn(qkv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, qkv_out_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_theta": 10000.0,
        "softcap": 30.0,
        "rms_norm_eps": 1e-6
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_theta: float,
    softcap: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 256
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    # Fuse Q/K/V projection into one matrix multiplication.
    query_states, key_states, value_states = F.linear(
        hidden_states, torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)
    ).split((num_attention_heads * head_dim, head_dim, head_dim), dim=-1)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Q RMSNorm
    query_states = F.rms_norm(query_states, (head_dim,), q_norm_weight, rms_norm_eps)
    
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # K RMSNorm
    key_states = F.rms_norm(key_states, (head_dim,), k_norm_weight, rms_norm_eps)
    
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # V RMSNorm (without scale)
    value_states = F.rms_norm(value_states, (head_dim,), None, rms_norm_eps)
    
    # Compute RoPE embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    cos = freqs.cos().to(hidden_states.dtype).unsqueeze(2)
    sin = freqs.sin().to(hidden_states.dtype).unsqueeze(2)
    
    # Apply RoPE to Q
    q1 = query_states[..., :head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    query_states = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
    
    # Apply RoPE to K
    k1 = key_states[..., :head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    key_states = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
    
    # Transpose for attention: [batch, heads, seq, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Store KV states for sharing before repeat
    key_states_out = key_states
    value_states_out = value_states
    
    # Keep the single KV head and let matmul broadcast it across query heads.
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    # Apply scaling and soft-capping in one pointwise expression.
    attn_weights = torch.tanh(attn_weights / (math.sqrt(head_dim) * softcap)) * softcap
    
    # Apply attention mask
    attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output, key_states_out, value_states_out
