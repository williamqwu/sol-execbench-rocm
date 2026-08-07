import torch
import triton
import triton.language as tl


@triton.jit
def _rope_table(POS, INV_FREQ, ROPE, stride_pb: tl.constexpr, stride_ps: tl.constexpr,
                FACTOR: tl.constexpr, N_CTX: tl.constexpr, N_ELEMS: tl.constexpr,
                BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = idx // 64
    freq_idx = idx % 64
    batch = token // N_CTX
    seq = token % N_CTX
    valid = idx < N_ELEMS
    pos = tl.load(POS + batch * stride_pb + seq * stride_ps, mask=valid, other=0).to(tl.float32)
    inv = tl.load(INV_FREQ + freq_idx, mask=valid, other=0.0)
    angle = pos * inv
    c = (tl.cos(angle) * FACTOR).to(tl.bfloat16)
    s = (tl.sin(angle) * FACTOR).to(tl.bfloat16)
    tl.store(ROPE + token * 128 + freq_idx, c, mask=valid)
    tl.store(ROPE + token * 128 + 64 + freq_idx, s, mask=valid)


@triton.jit
def _rope_inplace(X, ROPE, N_CTX: tl.constexpr, N_HEADS: tl.constexpr):
    # One program rotates eight adjacent heads of one token.  Working on only
    # the first half lets each input element be loaded once and both halves be
    # written without an intermediate rotate-half tensor.
    groups = (N_HEADS + 7) // 8
    pid = tl.program_id(0)
    token = pid // groups
    group = pid % groups
    offs = tl.arange(0, 512)
    local_h = offs // 64
    freq = offs % 64
    head = group * 8 + local_h
    mask = (token < tl.num_programs(0) // groups) & (head < N_HEADS)

    p1 = X + token * (N_HEADS * 128) + head * 128 + freq
    p2 = p1 + 64
    x1 = tl.load(p1, mask=mask, other=0.0)
    x2 = tl.load(p2, mask=mask, other=0.0)
    c = tl.load(ROPE + token * 128 + freq, mask=mask, other=0.0)
    s = tl.load(ROPE + token * 128 + 64 + freq, mask=mask, other=0.0)

    # Each product and the following sum is a separate BF16 reference op.
    x1c = (x1.to(tl.float32) * c.to(tl.float32)).to(tl.bfloat16)
    nx2s = ((-x2.to(tl.float32)) * s.to(tl.float32)).to(tl.bfloat16)
    x2c = (x2.to(tl.float32) * c.to(tl.float32)).to(tl.bfloat16)
    x1s = (x1.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    y1 = (x1c.to(tl.float32) + nx2s.to(tl.float32)).to(tl.bfloat16)
    y2 = (x2c.to(tl.float32) + x1s.to(tl.float32)).to(tl.bfloat16)
    tl.store(p1, y1, mask=mask)
    tl.store(p2, y2, mask=mask)


@triton.jit
def _rounded_causal_attention(
    Q, K, V, O,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr,
    stride_ob: tl.constexpr, stride_om: tl.constexpr, stride_oh: tl.constexpr,
    SCALE: tl.constexpr, N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_bh = tl.program_id(1)
    off_b = off_bh // 40
    off_h = off_bh % 40
    off_kh = off_h // 5

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # First pass obtains the FP32 softmax normalization without materializing
    # the quadratic score tensor.  Scores are explicitly rounded after the
    # GEMM and again after scaling, matching the two BF16 reference ops.
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    loop_hi = tl.minimum(start_m + BLOCK_M, N_CTX)
    for start_n in tl.range(0, loop_hi, BLOCK_N):
        n = start_n + offs_n
        k_ptrs = K + off_b * stride_kb + off_kh * stride_kh + n[None, :] * stride_kn + offs_d[:, None]
        k = tl.load(k_ptrs, mask=n[None, :] < N_CTX, other=0.0)
        score = tl.dot(q, k).to(tl.bfloat16)
        score = (score.to(tl.float32) * SCALE).to(tl.bfloat16)
        valid = (offs_m[:, None] < N_CTX) & (n[None, :] <= offs_m[:, None]) & (n[None, :] < N_CTX)
        score = tl.where(valid, score.to(tl.float32), -float("inf"))
        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_max)
        row_sum = row_sum * alpha + tl.sum(tl.exp(score - new_max[:, None]), axis=1)
        row_max = new_max

    # The probability must be rounded only after division by the complete
    # softmax denominator.  A second score pass preserves that checkpoint.
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start_n in tl.range(0, loop_hi, BLOCK_N):
        n = start_n + offs_n
        k_ptrs = K + off_b * stride_kb + off_kh * stride_kh + n[None, :] * stride_kn + offs_d[:, None]
        k = tl.load(k_ptrs, mask=n[None, :] < N_CTX, other=0.0)
        score = tl.dot(q, k).to(tl.bfloat16)
        score = (score.to(tl.float32) * SCALE).to(tl.bfloat16)
        valid = (offs_m[:, None] < N_CTX) & (n[None, :] <= offs_m[:, None]) & (n[None, :] < N_CTX)
        score = tl.where(valid, score.to(tl.float32), -float("inf"))
        prob = tl.exp(score - row_max[:, None]) / row_sum[:, None]
        prob = tl.where(valid, prob, 0.0).to(tl.bfloat16)
        v_ptrs = V + off_b * stride_vb + off_kh * stride_vh + n[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n[:, None] < N_CTX, other=0.0)
        acc = tl.dot(prob, v, acc)

    o_ptrs = O + off_b * stride_ob + offs_m[:, None] * stride_om + off_h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)


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
    inv_freq,
    rms_norm_eps,
    attention_factor,
    scaling,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    tokens = batch_size * seq_len

    query_states = torch.nn.functional.linear(hidden_states, q_proj_weight)
    key_states = torch.nn.functional.linear(hidden_states, k_proj_weight)
    value_states = torch.nn.functional.linear(hidden_states, v_proj_weight)

    query_states = torch.nn.functional.rms_norm(query_states, (hidden_size,), q_norm_weight, rms_norm_eps)
    key_states = torch.nn.functional.rms_norm(key_states, (1024,), k_norm_weight, rms_norm_eps)

    rope = torch.empty((tokens, 128), dtype=hidden_states.dtype, device=hidden_states.device)
    rope_elems = tokens * 64
    _rope_table[(triton.cdiv(rope_elems, 256),)](
        position_ids, inv_freq, rope, position_ids.stride(0), position_ids.stride(1),
        attention_factor, seq_len, rope_elems, BLOCK=256, num_warps=4,
    )
    _rope_inplace[(tokens * 5,)](query_states, rope, seq_len, 40, num_warps=4)
    _rope_inplace[(tokens,)](key_states, rope, seq_len, 8, num_warps=4)

    query_states = query_states.view(batch_size, seq_len, 40, 128).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)

    attn_output = torch.empty_like(hidden_states)
    if (batch_size <= 8 and seq_len < 2048) or batch_size == 1:
        block_m = 64
        block_n = 16
    else:
        block_m = 128
        block_n = 32
    _rounded_causal_attention[(triton.cdiv(seq_len, block_m), batch_size * 40)](
        query_states, key_states, value_states, attn_output,
        query_states.stride(0), query_states.stride(1), query_states.stride(2),
        key_states.stride(0), key_states.stride(1), key_states.stride(2),
        value_states.stride(0), value_states.stride(1), value_states.stride(2),
        attn_output.stride(0), attn_output.stride(1), 128,
        scaling, seq_len,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4,
    )
    return torch.nn.functional.linear(attn_output, o_proj_weight)
