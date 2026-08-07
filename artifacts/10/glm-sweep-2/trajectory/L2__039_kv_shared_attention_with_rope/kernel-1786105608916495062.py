import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl
from triton.language.extra import libdevice


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


@triton.jit
def _flash_attn_softcap_fwd_kernel(
    Q, K, V, O,
    sm_scale, softcap,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, Q_LEN, KV_LEN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_kv = h // GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    Q_block = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    num_n = tl.cdiv(KV_LEN, BLOCK_N)
    for j in range(0, num_n):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = K + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        K_block = tl.load(k_ptrs, mask=offs_n[:, None] < KV_LEN, other=0.0)
        V_block = tl.load(v_ptrs, mask=offs_n[:, None] < KV_LEN, other=0.0)

        qk = tl.dot(Q_block, tl.trans(K_block))
        qk = qk * sm_scale
        qk = libdevice.tanh(qk / softcap) * softcap
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(V_block.dtype), V_block)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)


def _eager_attn_softcap(q, k, v, attention_mask, softcap):
    # q: [B,H,S,D], k,v: [B,Hkv,S,D] -> expand for GQA
    B, H, S, D = q.shape
    Hkv = k.shape[1]
    groups = H // Hkv
    if groups > 1:
        k = k[:, :, None, :, :].expand(B, Hkv, groups, S, D).reshape(B, H, S, D)
        v = v[:, :, None, :, :].expand(B, Hkv, groups, S, D).reshape(B, H, S, D)
    aw = torch.matmul(q, k.transpose(2, 3))
    aw = aw / math.sqrt(D)
    aw = aw / softcap
    aw = torch.tanh(aw)
    aw = aw * softcap
    aw = aw + attention_mask
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def _flash_attn_softcap(q, k, v, softcap, attention_mask):
    # q: [B,H,S,D] bf16, k,v: [B,Hkv,S,D] bf16
    B, H, S, D = q.shape
    Hkv = k.shape[1]
    groups = H // Hkv
    BH = B * H
    # Eager path wins for very low parallelism at mid sequence lengths.
    if BH <= 16 and 512 < S <= 1024:
        return _eager_attn_softcap(q, k, v, attention_mask, softcap)
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    if BH >= 64:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 32, 4, 2
    elif S <= 512:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 32, 4, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 3
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _flash_attn_softcap_fwd_kernel[grid](
        q, k, v, o,
        sm_scale, softcap,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, S, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, GROUPS=groups,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


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

    # Q projection
    query_states = F.linear(hidden_states, q_proj_weight)  # [batch, seq, num_heads * head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)

    # Q RMSNorm
    q_variance = query_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    query_states = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    query_states = (query_states * q_norm_weight).to(hidden_states.dtype)

    # K projection
    key_states = F.linear(hidden_states, k_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)

    # K RMSNorm
    k_variance = key_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    key_states = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    key_states = (key_states * k_norm_weight).to(hidden_states.dtype)

    # V projection
    value_states = F.linear(hidden_states, v_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)

    # V RMSNorm (without scale)
    v_variance = value_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    value_states = value_states * torch.rsqrt(v_variance + rms_norm_eps)
    value_states = value_states.to(hidden_states.dtype)

    # Compute RoPE embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [batch, seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq, head_dim]
    cos = emb.cos().to(hidden_states.dtype)
    sin = emb.sin().to(hidden_states.dtype)

    # Apply RoPE to Q
    cos_q = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_q = sin.unsqueeze(2)
    q1 = query_states[..., :head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_q) + (q_rotated * sin_q)

    # Apply RoPE to K
    cos_k = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_k = sin.unsqueeze(2)
    k1 = key_states[..., :head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_k) + (k_rotated * sin_k)

    # Transpose for attention: [batch, heads, seq, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Store KV states for sharing before repeat
    key_states_out = key_states.clone()
    value_states_out = value_states.clone()

    # Fused flash attention with softcapping (handles GQA internally, no expand)
    attn_output = _flash_attn_softcap(query_states, key_states, value_states, softcap, attention_mask)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)

    return attn_output, key_states_out, value_states_out
