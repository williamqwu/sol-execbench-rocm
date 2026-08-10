import torch
import torch.nn.functional as F
import math


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


@torch.compile
def add_rms_norm(
    residual: torch.Tensor, update: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    x = residual + update
    return x, rms_norm(x, weight, eps)


@torch.compile
def mlp_tail(
    gate: torch.Tensor,
    up: torch.Tensor,
    down_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual + F.linear(F.silu(gate) * up, down_weight)


@torch.compile
def normalized_qkv(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = rms_norm(x, norm_weight, eps)
    return F.linear(x, q_weight), F.linear(x, k_weight), F.linear(x, v_weight)


@torch.compile
def add_norm_gate_up(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, normalized = add_rms_norm(residual, update, norm_weight, eps)
    return x, F.linear(normalized, gate_weight), F.linear(normalized, up_weight)


@torch.compile
def add_norm_query(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_weight: torch.Tensor,
    q_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, normalized = add_rms_norm(residual, update, norm_weight, eps)
    return x, F.linear(normalized, q_weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


@torch.compile
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> tuple:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    freqs = torch.outer(torch.arange(q.shape[2], device=q.device, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    
    batch_size, num_heads, seq_len, hd = q.shape
    cos = cos.view(1, 1, seq_len, hd)
    sin = sin.view(1, 1, seq_len, hd)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    self_attn_norm_weight: torch.Tensor,
    self_attn_q_weight: torch.Tensor,
    self_attn_k_weight: torch.Tensor,
    self_attn_v_weight: torch.Tensor,
    self_attn_o_weight: torch.Tensor,
    cross_attn_norm_weight: torch.Tensor,
    cross_attn_q_weight: torch.Tensor,
    cross_attn_k_weight: torch.Tensor,
    cross_attn_v_weight: torch.Tensor,
    cross_attn_o_weight: torch.Tensor,
    mlp_norm_weight: torch.Tensor,
    mlp_gate_weight: torch.Tensor,
    mlp_up_weight: torch.Tensor,
    mlp_down_weight: torch.Tensor,
    norm_eps: float,
    rope_theta: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    encoder_seq_len = encoder_hidden_states.shape[1]
    
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 128
    cross_num_attention_heads = 16
    cross_num_key_value_heads = 16
    cross_head_dim = 128
    
    residual = hidden_states
    
    # Self-Attention Block
    query_states, key_states, value_states = normalized_qkv(
        hidden_states,
        self_attn_norm_weight,
        self_attn_q_weight,
        self_attn_k_weight,
        self_attn_v_weight,
        norm_eps,
    )
    
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    query_states, key_states = apply_rope(query_states, key_states, head_dim, rope_theta)
    
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, self_attn_o_weight)
    
    residual, query_states = add_norm_query(
        residual,
        attn_output,
        cross_attn_norm_weight,
        cross_attn_q_weight,
        norm_eps,
    )
    
    # Cross-Attention Block
    
    key_states = F.linear(encoder_hidden_states, cross_attn_k_weight)
    value_states = F.linear(encoder_hidden_states, cross_attn_v_weight)
    
    query_states = query_states.view(batch_size, seq_len, cross_num_attention_heads, cross_head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    
    attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, cross_num_attention_heads * cross_head_dim)
    attn_output = F.linear(attn_output, cross_attn_o_weight)
    
    residual, gate, up = add_norm_gate_up(
        residual,
        attn_output,
        mlp_norm_weight,
        mlp_gate_weight,
        mlp_up_weight,
        norm_eps,
    )
    
    # MLP Block
    
    output = mlp_tail(gate, up, mlp_down_weight, residual)
    
    return output
