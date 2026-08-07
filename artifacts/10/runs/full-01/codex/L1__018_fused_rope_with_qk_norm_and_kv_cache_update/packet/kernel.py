import torch
import triton
import triton.language as tl

HEAD_GROUP = 4


@triton.jit
def _native_sincos(x):
    turns = x * 0.15915494309189535
    turns = turns - tl.floor(turns + 0.5)
    cos_turns = turns + 0.25
    cos_turns = cos_turns - tl.floor(cos_turns + 0.5)
    sinv, cosv = tl.inline_asm_elementwise(
        "v_sin_f32 $0, $2\n\tv_sin_f32 $1, $3",
        "=v,=v,v,v",
        [turns, cos_turns],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return sinv, cosv


@triton.jit
def _rope_trig_kernel(
    position_ids,
    inv_freq,
    trig,
    seq_len: tl.constexpr,
    pos_batch_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // seq_len
    token = pid - batch * seq_len
    d = tl.arange(0, 64)
    pos = tl.load(position_ids + batch * pos_batch_stride + token).to(tl.float32)
    angle = pos * tl.load(inv_freq + d)
    sinv, cosv = _native_sincos(angle)
    row = pid * 128
    tl.store(trig + row + d, cosv.to(tl.bfloat16))
    tl.store(trig + row + 64 + d, sinv.to(tl.bfloat16))


@triton.jit
def _fused_qk_rope_cache_kernel(
    query,
    key,
    value,
    position_ids,
    key_cache,
    value_cache,
    q_weight,
    k_weight,
    inv_freq,
    trig,
    query_out,
    key_out,
    cache_position,
    seq_len: tl.constexpr,
    pos_batch_stride: tl.constexpr,
    eps: tl.constexpr,
    MAX_POS: tl.constexpr,
    GROUP: tl.constexpr,
    PRECOMPUTED_TRIG: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_batch = 96 // GROUP
    token_group = pid % (groups_per_batch * seq_len)
    batch = pid // (groups_per_batch * seq_len)
    q_group = token_group // seq_len
    token = token_group - q_group * seq_len

    head = q_group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, 64)
    row = ((batch * 96 + head[:, None]) * seq_len + token) * 128
    q0 = tl.load(query + row + d[None, :]).to(tl.float32)
    q1 = tl.load(query + row + 64 + d[None, :]).to(tl.float32)
    q_var = tl.sum(q0 * q0 + q1 * q1, axis=1) * (1.0 / 128.0)
    q_scale = tl.rsqrt(q_var + eps)[:, None]
    qw0 = tl.load(q_weight + d).to(tl.float32)
    qw1 = tl.load(q_weight + 64 + d).to(tl.float32)

    # The reference materializes RMSNorm and the trig tensors in BF16.
    qn0 = (q0 * q_scale * qw0).to(tl.bfloat16)
    qn1 = (q1 * q_scale * qw1).to(tl.bfloat16)
    if PRECOMPUTED_TRIG:
        trig_row = (batch * seq_len + token) * 128
        cosv = tl.load(trig + trig_row + d)
        sinv = tl.load(trig + trig_row + 64 + d)
    else:
        pos = tl.load(position_ids + batch * pos_batch_stride + token).to(tl.float32)
        angle = pos * tl.load(inv_freq + d)
        sinv, cosv = _native_sincos(angle)
        cosv = cosv.to(tl.bfloat16)
        sinv = sinv.to(tl.bfloat16)

    qnf0 = qn0.to(tl.float32)
    qnf1 = qn1.to(tl.float32)
    cosf = cosv.to(tl.float32)
    sinf = sinv.to(tl.float32)
    qo0 = qnf0 * cosf - qnf1 * sinf
    tl.store(query_out + row + d[None, :], qo0)
    qo1 = qnf1 * cosf + qnf0 * sinf
    tl.store(query_out + row + 64 + d[None, :], qo1)

    # Pair query heads 0..7 with the eight KV heads. This keeps the entire
    # operation in a single launch without redundant KV work.
    if q_group * GROUP < 8:
        krow = ((batch * 8 + head[:, None]) * seq_len + token) * 128
        k0 = tl.load(key + krow + d[None, :]).to(tl.float32)
        k1 = tl.load(key + krow + 64 + d[None, :]).to(tl.float32)
        k_var = tl.sum(k0 * k0 + k1 * k1, axis=1) * (1.0 / 128.0)
        k_scale = tl.rsqrt(k_var + eps)[:, None]
        kw0 = tl.load(k_weight + d).to(tl.float32)
        kw1 = tl.load(k_weight + 64 + d).to(tl.float32)
        kn0 = (k0 * k_scale * kw0).to(tl.bfloat16)
        kn1 = (k1 * k_scale * kw1).to(tl.bfloat16)
        knf0 = kn0.to(tl.float32)
        knf1 = kn1.to(tl.float32)
        ko0 = knf0 * cosf - knf1 * sinf
        ko1 = knf1 * cosf + knf0 * sinf
        tl.store(key_out + krow + d[None, :], ko0)
        tl.store(key_out + krow + 64 + d[None, :], ko1)

        cache_pos = tl.load(cache_position + token)
        cache_row = ((batch * 8 + head[:, None]) * MAX_POS + cache_pos) * 128
        tl.store(key_cache + cache_row + d[None, :], ko0)
        tl.store(key_cache + cache_row + 64 + d[None, :], ko1)
        tl.store(value_cache + cache_row + d[None, :], tl.load(value + krow + d[None, :]))
        tl.store(value_cache + cache_row + 64 + d[None, :], tl.load(value + krow + 64 + d[None, :]))


@torch.no_grad()
def run(
    query,
    key,
    value,
    position_ids,
    key_cache,
    value_cache,
    cache_position,
    q_norm_weight,
    k_norm_weight,
    inv_freq,
    rms_norm_eps,
):
    batch_size, _, seq_len, _ = query.shape
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    group = HEAD_GROUP
    precomputed_trig = False
    if precomputed_trig:
        trig = torch.empty(
            (batch_size, seq_len, 128), device=query.device, dtype=query.dtype
        )
        _rope_trig_kernel[(batch_size * seq_len,)](
            position_ids,
            inv_freq,
            trig,
            seq_len=seq_len,
            pos_batch_stride=position_ids.stride(0),
            num_warps=1,
        )
    else:
        trig = query_out
    grid = (batch_size * (96 // group) * seq_len,)
    _fused_qk_rope_cache_kernel[grid](
        query,
        key,
        value,
        position_ids,
        key_cache,
        value_cache,
        q_norm_weight,
        k_norm_weight,
        inv_freq,
        trig,
        query_out,
        key_out,
        cache_position,
        seq_len=seq_len,
        pos_batch_stride=position_ids.stride(0),
        eps=float(rms_norm_eps),
        MAX_POS=key_cache.shape[2],
        GROUP=group,
        PRECOMPUTED_TRIG=precomputed_trig,
        num_warps=1,
    )
    return query_out, key_out, key_cache, value_cache
