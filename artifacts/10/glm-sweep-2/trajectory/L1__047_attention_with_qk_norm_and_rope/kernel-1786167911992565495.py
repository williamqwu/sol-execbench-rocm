import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def rms_norm(x, weight, eps):
    """RMSNorm with (1 + weight) scaling as used in Gemma3."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    output = x_normed * (1.0 + weight.float())
    return output.type_as(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    scale, softcap,
    B, H, Hkv, Sq, Sk, D,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    off_hkv = off_h // GROUPS

    Q_base = Q + off_b * stride_qb + off_h * stride_qh
    K_base = K + off_b * stride_kb + off_hkv * stride_kh
    V_base = V + off_b * stride_vb + off_hkv * stride_vh
    O_base = O + off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < Sq, other=0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, N_BLOCKS):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_n[:, None] < Sk, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = s.to(tl.float32)

        # softcap: tanh(s/cap)*cap
        sc = s / softcap
        sc = tl.where(sc > 20.0, 1.0, 1.0 - 2.0 / (tl.exp(2.0 * sc) + 1.0))
        sc = tl.where(sc < -20.0, -1.0, sc)
        s = sc * softcap

        if USE_CAUSAL:
            valid = offs_m[:, None] >= offs_n[None, :]
        else:
            valid = offs_n[None, :] < Sk
        s = tl.where(valid, s, -1e30)

        m_ij = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(s - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < Sk, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    tl.store(O_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < Sq)


def _flash_attention(query_states, key_states, value_states, scale, softcap, causal=True):
    B, H, Sq, D = query_states.shape
    _, Hkv, Sk, _ = key_states.shape
    groups = H // Hkv
    o = torch.empty(B, H, Sq, D, dtype=query_states.dtype, device=query_states.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D
    n_blocks = triton.cdiv(Sk, BLOCK_N)
    grid = (triton.cdiv(Sq, BLOCK_M), B * H)
    _flash_attn_fwd[grid](
        query_states, key_states, value_states, o,
        query_states.stride(0), query_states.stride(1), query_states.stride(2), query_states.stride(3),
        key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale, softcap,
        B, H, Hkv, Sq, Sk, D,
        GROUPS=groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        N_BLOCKS=n_blocks,
        USE_CAUSAL=causal,
        num_warps=8,
        num_stages=2,
    )
    return o


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5

    # Project to Q, K, V using F.linear
    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)

    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply Q/K normalization (unique to Gemma3)
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)

    # Apply RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Fused flash attention with softcap and causal masking (avoids materializing [B,H,S,S])
    attn_output = _flash_attention(
        query_states, key_states, value_states, scaling, attn_logit_softcapping, causal=True
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)

    return attn_output
