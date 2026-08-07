import torch
import torch.nn.functional as F
from typing import Tuple

# head_dim is constant (128); precompute the per-dimension RoPE inv_freq base once.
# rope_theta is a fixed runtime arg (500000.0) but the geometric spacing depends
# only on head_dim, so we build the base exponent vector at import time.
_HEAD_DIM = 128
_HALF = _HEAD_DIM // 2
_EXPONENT = torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32) / _HEAD_DIM


@torch.no_grad()
@torch.compile(dynamic=True, mode="default")
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    num_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    half = head_dim // 2

    bsz, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device

    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return weight * x.to(input_dtype)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    inv_freq = 1.0 / (rope_theta ** _EXPONENT.to(device))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    qf, kf = query_states.float(), key_states.float()
    q1, q2 = qf[..., :half], qf[..., half:]
    k1, k2 = kf[..., :half], kf[..., half:]
    query_states = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).to(query_states.dtype)
    key_states = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).to(key_states.dtype)

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, seq_len, -1)

    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)

    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(F.silu(gate) * up, down_proj_weight)
    hidden_states = residual + hidden_states

    return hidden_states
