import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
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
    # Constants
    num_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    half = head_dim // 2

    bsz, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device

    # RMSNorm: upcast to fp32, normalize, downcast. Compute variance in one pass.
    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return weight * x.to(input_dtype)

    # RoPE (rotate-half) computed via a cheap outer product; cos/sin shared by Q and K.
    def apply_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)          # [seq, half]
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1,1,seq,half]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        qf, kf = q.float(), k.float()
        q1, q2 = qf[..., :half], qf[..., half:]
        k1, k2 = kf[..., :half], kf[..., half:]
        q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    # Self-attention block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)

    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    # Reshape for multi-head attention
    query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    query_states, key_states = apply_rope(query_states, key_states)

    # Fused scaled-dot-product attention with GQA + causal mask.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, seq_len, -1)

    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output

    # MLP block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)

    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)
    hidden_states = residual + hidden_states

    return hidden_states
