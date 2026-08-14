import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _q_norm_rope(
    q_ptr,
    cos_ptr,
    sin_ptr,
    weight_ptr,
    eps,
):
    row = tl.program_id(0)
    d = tl.arange(0, 256)
    pair_d = tl.where(d < 128, d + 128, d - 128)
    base = row * 256

    x = tl.load(q_ptr + base + d).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 256.0)
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + d).to(tl.float32)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)

    pair_x = tl.load(q_ptr + base + pair_d).to(tl.float32)
    pair_weight = tl.load(weight_ptr + pair_d).to(tl.float32)
    pair_normalized = (pair_x * inv_rms * pair_weight).to(tl.bfloat16)
    rotated = tl.where(d < 128, -pair_normalized, pair_normalized)

    rope_row = row // 8
    cos = tl.load(cos_ptr + rope_row * 256 + d)
    sin = tl.load(sin_ptr + rope_row * 256 + d)
    lhs = (normalized.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
    rhs = (rotated.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
    result = (lhs.to(tl.float32) + rhs.to(tl.float32)).to(tl.bfloat16)
    tl.store(q_ptr + base + d, result)


@triton.jit
def _kv_norm_rope(
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    weight_ptr,
    position_ptr,
    position_stride_b: tl.constexpr,
    position_stride_s: tl.constexpr,
    rope_theta,
    eps,
    N_CTX: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 256)
    pair_d = tl.where(d < 128, d + 128, d - 128)
    base = row * 256

    k = tl.load(k_ptr + base + d).to(tl.float32)
    k_variance = tl.sum(k * k, axis=0) * (1.0 / 256.0)
    k_inv_rms = tl.rsqrt(k_variance + eps)
    weight = tl.load(weight_ptr + d).to(tl.float32)
    k_normalized = (k * k_inv_rms * weight).to(tl.bfloat16)

    pair_k = tl.load(k_ptr + base + pair_d).to(tl.float32)
    pair_weight = tl.load(weight_ptr + pair_d).to(tl.float32)
    pair_normalized = (pair_k * k_inv_rms * pair_weight).to(tl.bfloat16)
    k_rotated = tl.where(d < 128, -pair_normalized, pair_normalized)
    batch = row // N_CTX
    seq = row - batch * N_CTX
    position = tl.load(
        position_ptr + batch * position_stride_b + seq * position_stride_s
    ).to(tl.float32)
    freq_index = tl.where(d < 128, d, d - 128).to(tl.float32)
    exponent = freq_index * (1.0 / 128.0)
    inv_freq = 1.0 / libdevice.pow(rope_theta, exponent)
    angle = position * inv_freq
    cos = libdevice.cos(angle).to(tl.bfloat16)
    sin = libdevice.sin(angle).to(tl.bfloat16)
    tl.store(cos_ptr + base + d, cos)
    tl.store(sin_ptr + base + d, sin)
    k_lhs = (k_normalized.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
    k_rhs = (k_rotated.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
    k_result = (k_lhs.to(tl.float32) + k_rhs.to(tl.float32)).to(tl.bfloat16)
    tl.store(k_ptr + base + d, k_result)

    v = tl.load(v_ptr + base + d).to(tl.float32)
    v_variance = tl.sum(v * v, axis=0) * (1.0 / 256.0)
    v_inv_rms = tl.rsqrt(v_variance + eps)
    v_result = (v * v_inv_rms).to(tl.bfloat16)
    tl.store(v_ptr + base + d, v_result)


@triton.jit
def _shared_kv_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_mb: tl.constexpr,
    softcap,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // 8
    head = bh - batch * 8

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 256)

    q_offsets = (
        batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < N_CTX, other=0.0)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 256), tl.float32)

    # All workloads use the supplied causal mask.  Stop at this query tile,
    # while still applying the actual additive mask within the visited tiles.
    end_n = tl.minimum((block_m + 1) * BLOCK_M, N_CTX)
    for start_n in tl.range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        k_offsets = (
            batch * stride_kb
            + offs_n[:, None] * stride_ks
            + offs_d[None, :]
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=offs_n[:, None] < N_CTX,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        # torch.matmul writes bf16 here.  Preserve each subsequent bf16
        # elementwise rounding point from the reference implementation.
        scores = scores.to(tl.bfloat16).to(tl.float32) * 0.0625
        scores = (scores / softcap).to(tl.bfloat16).to(tl.float32)
        scores = libdevice.tanh(scores).to(tl.bfloat16).to(tl.float32)
        scores = (scores * softcap).to(tl.bfloat16).to(tl.float32)

        mask_offsets = (
            batch * stride_mb + offs_m[:, None] * N_CTX + offs_n[None, :]
        )
        additive_mask = tl.load(
            mask_ptr + mask_offsets,
            mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
            other=0.0,
        ).to(tl.float32)
        scores = (scores + additive_mask).to(tl.bfloat16).to(tl.float32)
        scores = tl.where(
            (offs_n[None, :] < N_CTX) & (offs_n[None, :] <= offs_m[:, None]),
            scores,
            -float("inf"),
        )

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        alpha = tl.exp(row_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        new_sum = row_sum * alpha + tl.sum(probs, axis=1)

        v = tl.load(
            v_ptr + k_offsets,
            mask=offs_n[:, None] < N_CTX,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(
            probs.to(tl.bfloat16), v, out_dtype=tl.float32
        )
        row_max = new_max
        row_sum = new_sum

    acc = acc / row_sum[:, None]
    out_offsets = (
        batch * (N_CTX * 8 * 256)
        + offs_m[:, None] * (8 * 256)
        + head * 256
        + offs_d[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=offs_m[:, None] < N_CTX,
    )


@torch.no_grad()
def run(
    hidden_states,
    position_ids,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    rope_theta,
    softcap,
    rms_norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = 256

    query_states = F.linear(hidden_states, q_proj_weight).view(
        batch_size, seq_len, 8, head_dim
    )

    key_states = F.linear(hidden_states, k_proj_weight).view(
        batch_size, seq_len, 1, head_dim
    )

    value_states = F.linear(hidden_states, v_proj_weight).view(
        batch_size, seq_len, 1, head_dim
    )

    cos = torch.empty(
        (batch_size, seq_len, head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    sin = torch.empty_like(cos)
    _kv_norm_rope[(batch_size * seq_len,)](
        key_states,
        value_states,
        cos,
        sin,
        k_norm_weight,
        position_ids,
        position_ids.stride(0),
        position_ids.stride(1),
        rope_theta,
        rms_norm_eps,
        N_CTX=seq_len,
        num_warps=4,
    )
    _q_norm_rope[(batch_size * seq_len * 8,)](
        query_states,
        cos,
        sin,
        q_norm_weight,
        rms_norm_eps,
        num_warps=4,
    )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    key_states_out = key_states.clone()
    value_states_out = value_states.clone()

    attn_output = torch.empty(
        (batch_size, seq_len, 8, head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Larger query tiles amortize K/V traffic once the grid has ample
    # batch/head parallelism; retain 64 rows for low-batch occupancy.
    block_m = (
        128 if 8 <= batch_size < 32 and seq_len >= 512 else 64
    )
    block_n = 64
    _shared_kv_attention[(triton.cdiv(seq_len, block_m), batch_size * 8)](
        query_states,
        key_states,
        value_states,
        attention_mask,
        attn_output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(2),
        attention_mask.stride(0),
        softcap,
        N_CTX=seq_len,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    attn_output = attn_output.reshape(batch_size, seq_len, 8 * head_dim)
    attn_output = F.linear(attn_output, o_proj_weight)
    return attn_output, key_states_out, value_states_out
