import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as _libdevice_tanh


def rms_norm(x, weight, eps):
    """RMSNorm with (1 + weight) scaling as used in Gemma3."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    output = x_normed * (1.0 + weight.float())
    return output.type_as(x)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Triton flash-attention forward with causal mask + tanh logit softcapping.
# Standard online-softmax (flash-attention) algorithm.
# Q/K/V layout: [B, H, S, D]. Ouput: [B, H, S, D].
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,
    sm_scale,
    softcap,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, H_KV, S_Q, S_KV, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GQA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hb = tl.program_id(1)

    pid_h = pid_hb % H
    pid_b = pid_hb // H
    pid_h_kv = pid_h // GQA

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptrs = K + pid_b * stride_kb + pid_h_kv * stride_kh
    V_ptrs = V + pid_b * stride_vb + pid_h_kv * stride_vh

    Q_block = Q_ptrs + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(Q_block, mask=offs_m[:, None] < S_Q, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # causal: only attend to keys j <= i. End at the diagonal block of the largest query row.
    end_m = tl.minimum(pid_m * BLOCK_M + BLOCK_M, S_Q)
    end_n = tl.minimum(end_m, S_KV)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        K_block = K_ptrs + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        V_block = V_ptrs + offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(K_block, mask=offs_n_curr[:, None] < S_KV, other=0.0).to(tl.float32)
        v = tl.load(V_block, mask=offs_n_curr[:, None] < S_KV, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # softcapping: tanh(qk / softcap) * softcap
        qk = qk / softcap
        qk = _libdevice_tanh(qk)
        qk = qk * softcap

        # causal mask
        mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(mask, qk, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # guard against -inf (rows fully masked -> m stays -inf)
        m_ij_safe = tl.where(m_ij == float('-inf'), 0.0, m_ij)
        alpha = tl.exp2(m_i - m_ij_safe)
        p = tl.exp2(qk - m_ij_safe[:, None])
        p = tl.where(mask, p, 0.0)

        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Note: we used exp2 which means scale must be log2(e)*sm_scale for true softmax.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    O_ptrs = O + pid_b * stride_ob + pid_h * stride_oh
    O_block = O_ptrs + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(O_block, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S_Q)


def _flash_attn(q, k, v, sm_scale, softcap):
    """q: [B,H,S,D], k/v: [B,H_kv,S,D]. Returns o: [B,H,S,D]."""
    B, H, S_Q, D = q.shape
    H_KV = k.shape[1]
    S_KV = k.shape[2]
    GQA = H // H_KV
    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S_Q, BLOCK_M), B * H)
    _flash_attn_fwd[grid](
        q, k, v, o,
        sm_scale, softcap,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, H_KV, S_Q, S_KV, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GQA=GQA,
        num_warps=4, num_stages=2,
    )
    return o


# softmax scale: the reference does softmax in fp32 of (qk*scale) then tanh-softcap.
# Our flash kernel computes p = exp2(qk*log2e*scale ...) after softcap.
# exp2(x) = e^x, so to get exp(qk...) we pass sm_scale*log2(e).
import math
_LOG2E = math.log2(math.e)


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

    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)

    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Flash attention with GQA (no repeat_kv needed - kernel reads KV per group)
    attn_output = _flash_attn(
        query_states, key_states, value_states,
        scaling * _LOG2E,  # convert e^x base to 2^x base
        float(attn_logit_softcapping),
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    attn_output = F.linear(attn_output, o_proj_weight)

    return attn_output
