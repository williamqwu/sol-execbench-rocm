import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_tunable = torch.cuda.tunable
_tunable.set_filename(os.path.join(os.path.dirname(__file__), "tunable.csv"))
_tunable.tuning_enable(False)
_tunable.write_file_on_exit(False)


@triton.jit
def _q_norm_rope_reorder(
    q_ptr,
    q_out_ptr,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    eps,
    SEQ_LEN: tl.constexpr,
):
    # Destination rows are [batch, kv_head, token, q_group, dim].
    row = tl.program_id(0)
    group = row % 12
    token_row = row // 12
    token = token_row % SEQ_LEN
    kv_head = (token_row // SEQ_LEN) % 8
    batch = token_row // (SEQ_LEN * 8)
    src_row = (batch * SEQ_LEN + token) * 96 + kv_head * 12 + group

    half = tl.arange(0, 64)
    x1 = tl.load(q_ptr + src_row * 128 + half).to(tl.float32)
    x2 = tl.load(q_ptr + src_row * 128 + 64 + half).to(tl.float32)
    variance = tl.sum(x1 * x1 + x2 * x2, axis=0) * (1.0 / 128.0)
    inv = tl.rsqrt(variance + eps)

    w1 = tl.load(norm_ptr + half).to(tl.float32)
    w2 = tl.load(norm_ptr + 64 + half).to(tl.float32)
    n1 = (w1 * (x1 * inv)).to(tl.bfloat16)
    n2 = (w2 * (x2 * inv)).to(tl.bfloat16)

    pos = (batch * SEQ_LEN + token) * 128
    c1 = tl.load(cos_ptr + pos + half).to(tl.float32)
    c2 = tl.load(cos_ptr + pos + 64 + half).to(tl.float32)
    s1 = tl.load(sin_ptr + pos + half).to(tl.float32)
    s2 = tl.load(sin_ptr + pos + 64 + half).to(tl.float32)

    a1 = (n1.to(tl.float32) * c1).to(tl.bfloat16)
    b1 = ((-n2).to(tl.float32) * s1).to(tl.bfloat16)
    a2 = (n2.to(tl.float32) * c2).to(tl.bfloat16)
    b2 = (n1.to(tl.float32) * s2).to(tl.bfloat16)
    y1 = (a1.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    y2 = (a2.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    tl.store(q_out_ptr + row * 128 + half, y1)
    tl.store(q_out_ptr + row * 128 + 64 + half, y2)


@triton.jit
def _k_norm_rope_v_reorder(
    k_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    eps,
    SEQ_LEN: tl.constexpr,
):
    # Destination rows are [batch, kv_head, token, dim].
    row = tl.program_id(0)
    token = row % SEQ_LEN
    group_row = row // SEQ_LEN
    kv_head = group_row % 8
    batch = group_row // 8
    src_row = (batch * SEQ_LEN + token) * 8 + kv_head

    half = tl.arange(0, 64)
    x1 = tl.load(k_ptr + src_row * 128 + half).to(tl.float32)
    x2 = tl.load(k_ptr + src_row * 128 + 64 + half).to(tl.float32)
    variance = tl.sum(x1 * x1 + x2 * x2, axis=0) * (1.0 / 128.0)
    inv = tl.rsqrt(variance + eps)

    w1 = tl.load(norm_ptr + half).to(tl.float32)
    w2 = tl.load(norm_ptr + 64 + half).to(tl.float32)
    n1 = (w1 * (x1 * inv)).to(tl.bfloat16)
    n2 = (w2 * (x2 * inv)).to(tl.bfloat16)

    pos = (batch * SEQ_LEN + token) * 128
    c1 = tl.load(cos_ptr + pos + half).to(tl.float32)
    c2 = tl.load(cos_ptr + pos + 64 + half).to(tl.float32)
    s1 = tl.load(sin_ptr + pos + half).to(tl.float32)
    s2 = tl.load(sin_ptr + pos + 64 + half).to(tl.float32)
    a1 = (n1.to(tl.float32) * c1).to(tl.bfloat16)
    b1 = ((-n2).to(tl.float32) * s1).to(tl.bfloat16)
    a2 = (n2.to(tl.float32) * c2).to(tl.bfloat16)
    b2 = (n1.to(tl.float32) * s2).to(tl.bfloat16)
    y1 = (a1.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    y2 = (a2.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    tl.store(k_out_ptr + row * 128 + half, y1)
    tl.store(k_out_ptr + row * 128 + 64 + half, y2)

    tl.store(v_out_ptr + row * 128 + half, tl.load(v_ptr + src_row * 128 + half))
    tl.store(v_out_ptr + row * 128 + 64 + half, tl.load(v_ptr + src_row * 128 + 64 + half))


@triton.jit
def _causal_scale_softmax(scores_ptr, SEQ_LEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    query_pos = (row // 12) % SEQ_LEN
    active = (col < SEQ_LEN) & (col <= query_pos)
    logits = tl.load(scores_ptr + row * SEQ_LEN + col, mask=active, other=-float("inf")).to(tl.float32)
    # The reference materializes this BF16 multiply before converting the
    # logits to FP32 for softmax.
    logits = (logits * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
    logits = logits - tl.max(logits, axis=0)
    numer = tl.exp(logits)
    probs = numer / tl.sum(numer, axis=0)
    tl.store(
        scores_ptr + row * SEQ_LEN + col,
        probs.to(tl.bfloat16),
        mask=col < SEQ_LEN,
    )


@torch.no_grad()
def run(
    hidden_states,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    rms_norm_eps,
):
    batch_size, seq_length, _ = hidden_states.shape
    token_count = batch_size * seq_length
    use_tunable = token_count != 128 and token_count != 1024
    if use_tunable:
        _tunable.enable(True)

    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    query_grouped = torch.empty_like(query_states)
    key_grouped = torch.empty_like(key_states)
    value_grouped = torch.empty_like(value_states)
    _q_norm_rope_reorder[(batch_size * seq_length * 96,)](
        query_states,
        query_grouped,
        q_norm_weight,
        cos,
        sin,
        rms_norm_eps,
        SEQ_LEN=seq_length,
        num_warps=1,
    )
    _k_norm_rope_v_reorder[(batch_size * seq_length * 8,)](
        key_states,
        value_states,
        key_grouped,
        value_grouped,
        k_norm_weight,
        cos,
        sin,
        rms_norm_eps,
        SEQ_LEN=seq_length,
        num_warps=1,
    )
    query_states = query_grouped.view(batch_size * 8, 12 * seq_length, 128)
    key_states = key_grouped.view(batch_size * 8, seq_length, 128)
    value_states = value_grouped.view(batch_size * 8, seq_length, 128)

    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    softmax_kernel = _causal_scale_softmax[(batch_size * 96 * seq_length,)]
    if seq_length == 2048:
        softmax_kernel(
            attn_weights,
            SEQ_LEN=seq_length,
            BLOCK=triton.next_power_of_2(seq_length),
            num_warps=1,
            waves_per_eu=8,
        )
    else:
        softmax_kernel(
            attn_weights,
            SEQ_LEN=seq_length,
            BLOCK=triton.next_power_of_2(seq_length),
            num_warps=1,
        )
    attn_output = torch.bmm(
        attn_weights, value_states
    )
    attn_output = (
        attn_output.view(batch_size, 8, seq_length, 12, 128)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(batch_size, seq_length, 96 * 128)
    )
    output = F.linear(attn_output, o_proj_weight, None)
    if use_tunable:
        _tunable.enable(False)
    return output
