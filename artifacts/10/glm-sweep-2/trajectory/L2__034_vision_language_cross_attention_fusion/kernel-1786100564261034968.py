import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    lang_seq_len = axes_and_scalars["lang_seq_len"]
    vision_seq_len = axes_and_scalars["vision_seq_len"]
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    
    language_hidden_states = torch.randn(batch_size, lang_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    vision_hidden_states = torch.randn(batch_size, vision_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    language_position_ids = torch.arange(lang_seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    
    vision_grid_thw = torch.zeros(batch_size, vision_seq_len, 3, dtype=torch.int64, device=device)
    for b in range(batch_size):
        for i in range(vision_seq_len):
            t = i // 196
            spatial_idx = i % 196
            h = spatial_idx // 14
            w = spatial_idx % 14
            vision_grid_thw[b, i, 0] = t
            vision_grid_thw[b, i, 1] = h
            vision_grid_thw[b, i, 2] = w
    
    q_proj_weight = torch.randn(num_attention_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    q_proj_bias = torch.randn(num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    return {
        "language_hidden_states": language_hidden_states,
        "vision_hidden_states": vision_hidden_states,
        "language_position_ids": language_position_ids,
        "vision_grid_thw": vision_grid_thw,
        "q_proj_weight": q_proj_weight,
        "q_proj_bias": q_proj_bias,
        "k_proj_weight": k_proj_weight,
        "k_proj_bias": k_proj_bias,
        "v_proj_weight": v_proj_weight,
        "v_proj_bias": v_proj_bias,
        "o_proj_weight": o_proj_weight,
    }


def _rope_1d(q, cos, sin):
    # q: (batch, num_heads, seq_len, head_dim), cos/sin: (batch, 1, seq_len, head_dim)
    x1 = q[..., : q.shape[-1] // 2]
    x2 = q[..., q.shape[-1] // 2 :]
    cos1 = cos[..., : q.shape[-1] // 2]
    sin1 = sin[..., : q.shape[-1] // 2]
    return torch.cat([x1 * cos1 - x2 * sin1, x2 * cos1 + x1 * sin1], dim=-1)


def _rope_component(x, pos, inv_freq):
    # x: (batch, num_heads, seq_len, dim), pos: (batch, seq_len), inv_freq: (dim//2,)
    pos_expanded = pos.unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, seq_len, 1)
    freqs = pos_expanded * inv_freq  # (batch, 1, seq_len, dim//2)
    cos = torch.cos(freqs).to(x.dtype)
    sin = torch.sin(freqs).to(x.dtype)
    return x * cos + _rotate_half_direct(x, sin)


def _rotate_half_direct(x, sin):
    # equivalent to rotate_half(x)*sin but computes directly
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2 * sin[..., : x.shape[-1] // 2], x1 * sin[..., : x.shape[-1] // 2]], dim=-1)


@torch.no_grad()
def run(
    language_hidden_states: torch.Tensor,
    vision_hidden_states: torch.Tensor,
    language_position_ids: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_kv_groups = num_attention_heads // num_key_value_heads
    rope_theta = 10000.0
    
    batch_size, lang_seq_len, _ = language_hidden_states.shape
    vision_seq_len = vision_hidden_states.shape[1]
    device = language_hidden_states.device
    
    inv_freq_1d = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    dim_t = 42
    dim_h = 42
    dim_w = 44
    inv_freq_t = 1.0 / (rope_theta ** (torch.arange(0, dim_t, 2, dtype=torch.float32, device=device) / dim_t))
    inv_freq_h = 1.0 / (rope_theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32, device=device) / dim_h))
    inv_freq_w = 1.0 / (rope_theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32, device=device) / dim_w))
    
    # Project queries from language tokens
    query_states = F.linear(language_hidden_states, q_proj_weight, q_proj_bias)
    query_states = query_states.view(batch_size, lang_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    key_states = F.linear(vision_hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(vision_hidden_states, v_proj_weight, v_proj_bias)
    
    key_states = key_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # 1D RoPE for query
    position_ids_expanded = language_position_ids.unsqueeze(1).unsqueeze(-1).float()
    freqs_1d = position_ids_expanded * inv_freq_1d
    emb_1d = torch.cat([freqs_1d, freqs_1d], dim=-1)
    cos_1d = emb_1d.cos().to(query_states.dtype)
    sin_1d = emb_1d.sin().to(query_states.dtype)
    q1 = query_states[..., :64]
    q2 = query_states[..., 64:]
    cos_1d_a = cos_1d[..., :64]
    sin_1d_a = sin_1d[..., :64]
    query_states = torch.cat([q1 * cos_1d_a - q2 * sin_1d_a, q2 * cos_1d_a + q1 * sin_1d_a], dim=-1)
    
    # 3D RoPE for key
    t_pos = vision_grid_thw[..., 0]
    h_pos = vision_grid_thw[..., 1]
    w_pos = vision_grid_thw[..., 2]
    
    k_t = key_states[..., :dim_t]
    k_h = key_states[..., dim_t:dim_t+dim_h]
    k_w = key_states[..., dim_t+dim_h:]
    
    # t component
    pos_t_exp = t_pos.unsqueeze(1).unsqueeze(-1).float()
    freqs_t = pos_t_exp * inv_freq_t
    cos_t = torch.cat([freqs_t, freqs_t], dim=-1).cos().to(k_t.dtype)
    sin_t = torch.cat([freqs_t, freqs_t], dim=-1).sin().to(k_t.dtype)
    k_t = _rope_1d(k_t, cos_t, sin_t)
    
    # h component
    pos_h_exp = h_pos.unsqueeze(1).unsqueeze(-1).float()
    freqs_h = pos_h_exp * inv_freq_h
    cos_h = torch.cat([freqs_h, freqs_h], dim=-1).cos().to(k_h.dtype)
    sin_h = torch.cat([freqs_h, freqs_h], dim=-1).sin().to(k_h.dtype)
    k_h = _rope_1d(k_h, cos_h, sin_h)
    
    # w component
    pos_w_exp = w_pos.unsqueeze(1).unsqueeze(-1).float()
    freqs_w = pos_w_exp * inv_freq_w
    cos_w = torch.cat([freqs_w, freqs_w], dim=-1).cos().to(k_w.dtype)
    sin_w = torch.cat([freqs_w, freqs_w], dim=-1).sin().to(k_w.dtype)
    k_w = _rope_1d(k_w, cos_w, sin_w)
    
    key_states = torch.cat([k_t, k_h, k_w], dim=-1)
    
    # Repeat k/v heads for GQA
    key_states = key_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_kv_groups, vision_seq_len, head_dim).reshape(batch_size, num_attention_heads, vision_seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_kv_groups, vision_seq_len, head_dim).reshape(batch_size, num_attention_heads, vision_seq_len, head_dim)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, lang_seq_len, num_attention_heads * head_dim)
    output = F.linear(attn_output, o_proj_weight)
    
    return output
