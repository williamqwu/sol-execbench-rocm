import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_rope_inplace(
    x_ptr, cos_ptr, sin_ptr, weight_ptr,
    x_stride_b: tl.constexpr, x_stride_h: tl.constexpr,
    x_stride_s: tl.constexpr, seq_len, heads: tl.constexpr, eps,
):
    vector = tl.program_id(0)
    vectors_per_batch = heads * seq_len
    batch = vector // vectors_per_batch
    within_batch = vector - batch * vectors_per_batch
    head = within_batch // seq_len
    position = within_batch - head * seq_len
    dims = tl.arange(0, 128)
    base = batch * x_stride_b + head * x_stride_h + position * x_stride_s

    x = tl.load(x_ptr + base + dims).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 128.0)
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + dims).to(tl.float32)
    normalized = (x * inv_rms * (1.0 + weight)).to(tl.bfloat16)

    partner_dims = tl.where(dims < 64, dims + 64, dims - 64)
    partner_x = tl.load(x_ptr + base + partner_dims).to(tl.float32)
    partner_weight = tl.load(weight_ptr + partner_dims).to(tl.float32)
    partner = (partner_x * inv_rms * (1.0 + partner_weight)).to(tl.bfloat16)
    rotated = tl.where(dims < 64, -partner, partner)

    rope_base = (batch * seq_len + position) * 128
    cosine = tl.load(cos_ptr + rope_base + dims)
    sine = tl.load(sin_ptr + rope_base + dims)
    first = (normalized * cosine).to(tl.bfloat16)
    second = (rotated * sine).to(tl.bfloat16)
    tl.store(x_ptr + base + dims, (first + second).to(tl.bfloat16))


def _norm_rope(x, cos, sin, weight, eps):
    batch, heads, seq_len, _ = x.shape
    _norm_rope_inplace[(batch * heads * seq_len,)](
        x, cos, sin, weight,
        x.stride(0), x.stride(1), x.stride(2), seq_len, heads, eps,
        num_warps=2,
    )
    return x


@triton.jit
def _streaming_attention(
    q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr, logits_ptr, block_max_ptr,
    q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
    q_stride_m: tl.constexpr, q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr, k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr, k_stride_d: tl.constexpr,
    mask_stride_b: tl.constexpr, mask_stride_m: tl.constexpr,
    out_stride_b: tl.constexpr, out_stride_h: tl.constexpr,
    out_stride_m: tl.constexpr, out_stride_d: tl.constexpr,
    seq_len, softcap,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CACHE_LOGITS: tl.constexpr,
    CACHE_EXP: tl.constexpr,
):
    block_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // 24
    q_head = bh - batch * 24
    kv_head = q_head // 3

    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 128)
    q = tl.load(
        q_ptr + batch * q_stride_b + q_head * q_stride_h
        + rows[:, None] * q_stride_m + dims[None, :] * q_stride_d,
        mask=rows[:, None] < seq_len,
        other=0.0,
    )

    neg_inf = float("-inf")
    row_max = tl.full((BLOCK_M,), neg_inf, tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    scale = 0.08838834764831845

    # Pass 1 computes the same global FP32 softmax normalizer as the reference.
    for start_n in range(0, seq_len, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + batch * k_stride_b + kv_head * k_stride_h
            + cols[None, :] * k_stride_n + dims[:, None] * k_stride_d,
            mask=cols[None, :] < seq_len,
            other=0.0,
        )
        scores = tl.dot(q, k).to(tl.bfloat16)
        scores = (scores * scale).to(tl.bfloat16)
        scores = (scores / softcap).to(tl.bfloat16)
        scores = libdevice.tanh(scores.to(tl.float32)).to(tl.bfloat16)
        scores = (scores * softcap).to(tl.bfloat16)
        additive_mask = tl.load(
            mask_ptr + batch * mask_stride_b
            + rows[:, None] * mask_stride_m + cols[None, :],
            mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
            other=neg_inf,
        )
        scores = (scores + additive_mask).to(tl.bfloat16).to(tl.float32)
        if CACHE_LOGITS and not CACHE_EXP:
            tl.store(
                logits_ptr + (bh * seq_len + rows[:, None]) * seq_len + cols[None, :],
                scores,
                mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
            )

        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        probabilities = tl.exp(scores - block_max[:, None])
        if CACHE_EXP:
            tl.store(
                logits_ptr + (bh * seq_len + rows[:, None]) * seq_len + cols[None, :],
                probabilities,
                mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
            )
            num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
            tl.store(
                block_max_ptr + (bh * num_n_blocks + start_n // BLOCK_N) * seq_len + rows,
                block_max,
                mask=rows < seq_len,
            )
        alpha = tl.exp(row_max - block_max)
        block_sum = tl.sum(probabilities, axis=1)
        row_sum = row_sum * alpha + block_sum
        row_max = block_max

    # Pass 2 rounds normalized probabilities to BF16 before the P@V dot.
    # That intermediate rounding is observable after the output projection.
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        if CACHE_EXP:
            cached_exp = tl.load(
                logits_ptr + (bh * seq_len + rows[:, None]) * seq_len + cols[None, :],
                mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
                other=0.0,
            ).to(tl.float32)
            num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
            cached_max = tl.load(
                block_max_ptr + (bh * num_n_blocks + start_n // BLOCK_N) * seq_len + rows,
                mask=rows < seq_len,
                other=neg_inf,
            )
            probabilities = (cached_exp * tl.exp(cached_max[:, None] - row_max[:, None])
                             / row_sum[:, None])
        elif CACHE_LOGITS:
            scores = tl.load(
                logits_ptr + (bh * seq_len + rows[:, None]) * seq_len + cols[None, :],
                mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
                other=neg_inf,
            ).to(tl.float32)
        else:
            k = tl.load(
                k_ptr + batch * k_stride_b + kv_head * k_stride_h
                + cols[None, :] * k_stride_n + dims[:, None] * k_stride_d,
                mask=cols[None, :] < seq_len,
                other=0.0,
            )
            scores = tl.dot(q, k).to(tl.bfloat16)
            scores = (scores * scale).to(tl.bfloat16)
            scores = (scores / softcap).to(tl.bfloat16)
            scores = libdevice.tanh(scores.to(tl.float32)).to(tl.bfloat16)
            scores = (scores * softcap).to(tl.bfloat16)
            additive_mask = tl.load(
                mask_ptr + batch * mask_stride_b
                + rows[:, None] * mask_stride_m + cols[None, :],
                mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
                other=neg_inf,
            )
            scores = (scores + additive_mask).to(tl.bfloat16).to(tl.float32)
        if not CACHE_EXP:
            probabilities = tl.exp(scores - row_max[:, None]) / row_sum[:, None]
        v = tl.load(
            v_ptr + batch * k_stride_b + kv_head * k_stride_h
            + cols[:, None] * k_stride_n + dims[None, :] * k_stride_d,
            mask=cols[:, None] < seq_len,
            other=0.0,
        )
        acc += tl.dot(probabilities.to(tl.bfloat16), v)

    out = acc
    tl.store(
        out_ptr + batch * out_stride_b + q_head * out_stride_h
        + rows[:, None] * out_stride_m + dims[None, :] * out_stride_d,
        out,
        mask=rows[:, None] < seq_len,
    )


def _attention(q, k, v, attention_mask, softcap):
    batch, _, seq_len, _ = q.shape
    # This allocation has B,S,H,D physical order, so the transpose after
    # attention is only a view (the reference's contiguous copy disappears).
    out_base = torch.empty((batch, seq_len, 24, 128), device=q.device, dtype=q.dtype)
    out = out_base.transpose(1, 2)
    cache_logits = True
    block_m = 64
    use_wide_tile = seq_len <= 128 or (384 <= seq_len <= 512)
    block_n = 128 if use_wide_tile else 64
    num_warps = 8 if use_wide_tile else 4
    # At very long context the extra HBM traffic of caching FP32 exponentials
    # is cheaper than evaluating exp a second time.
    cache_exp = seq_len >= 6000
    logits = torch.empty((batch, 24, seq_len, seq_len), device=q.device,
                         dtype=torch.float32 if cache_exp else q.dtype)
    block_maxes = (torch.empty((batch * 24, triton.cdiv(seq_len, block_n), seq_len),
                               device=q.device, dtype=torch.float32)
                   if cache_exp else q)
    _streaming_attention[(triton.cdiv(seq_len, block_m), batch * 24)](
        q, k, v, attention_mask, out, logits, block_maxes,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        attention_mask.stride(0), attention_mask.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len, softcap,
        BLOCK_M=block_m, BLOCK_N=block_n,
        CACHE_LOGITS=cache_logits,
        CACHE_EXP=cache_exp,
        num_warps=num_warps,
    )
    return out


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    attn_logit_softcapping,
    rms_norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape

    q = F.linear(hidden_states, q_proj_weight).view(batch_size, seq_len, 24, 128).transpose(1, 2)
    k = F.linear(hidden_states, k_proj_weight).view(batch_size, seq_len, 8, 128).transpose(1, 2)
    v = F.linear(hidden_states, v_proj_weight).view(batch_size, seq_len, 8, 128).transpose(1, 2)

    q = _norm_rope(q, cos, sin, q_norm_weight, rms_norm_eps)
    k = _norm_rope(k, cos, sin, k_norm_weight, rms_norm_eps)

    out = _attention(q, k, v, attention_mask, attn_logit_softcapping).transpose(1, 2).reshape(batch_size, seq_len, 3072)
    return F.linear(out, o_proj_weight)
