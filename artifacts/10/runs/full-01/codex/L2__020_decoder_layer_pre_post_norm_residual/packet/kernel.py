import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x, weight, output, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < 6144
    values = tl.load(x + row * 6144 + col, mask=valid, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) * (1.0 / 6144.0)
    normalized = (values * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    scale = tl.load(weight + col, mask=valid, other=0.0)
    tl.store(output + row * 6144 + col, normalized * scale, mask=valid)


@triton.jit
def _add_rms_norm_kernel(x, update, weight, residual, output, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < 6144
    x_value = tl.load(x + row * 6144 + col, mask=valid, other=0.0)
    update_value = tl.load(update + row * 6144 + col, mask=valid, other=0.0)
    # The residual addition is itself an eager bf16 operation in the reference.
    added = (x_value + update_value).to(tl.bfloat16)
    added_fp32 = added.to(tl.float32)
    variance = tl.sum(added_fp32 * added_fp32, axis=0) * (1.0 / 6144.0)
    normalized = (added_fp32 * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    scale = tl.load(weight + col, mask=valid, other=0.0)
    tl.store(residual + row * 6144 + col, added, mask=valid)
    tl.store(output + row * 6144 + col, normalized * scale, mask=valid)


@triton.jit
def _rope_query_kernel(query, cos, sin, output, TOTAL_ELEMENTS: tl.constexpr,
                       SEQ_LEN: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL_ELEMENTS
    d = index % 96
    rest = index // 96
    seq = rest % SEQ_LEN
    rest = rest // SEQ_LEN
    head = rest % 64
    batch = rest // 64
    half = d % 48
    source_base = ((batch * SEQ_LEN + seq) * 64 + head) * 96
    first = tl.load(query + source_base + half, mask=valid, other=0.0)
    second = tl.load(query + source_base + half + 48, mask=valid, other=0.0)
    trig_base = (batch * SEQ_LEN + seq) * 48 + half
    cos_value = tl.load(cos + trig_base, mask=valid, other=0.0)
    sin_value = tl.load(sin + trig_base, mask=valid, other=0.0)

    first_cos = (first * cos_value).to(tl.bfloat16)
    second_sin = (second * sin_value).to(tl.bfloat16)
    first_sin = (first * sin_value).to(tl.bfloat16)
    second_cos = (second * cos_value).to(tl.bfloat16)
    low = (first_cos - second_sin).to(tl.bfloat16)
    high = (first_sin + second_cos).to(tl.bfloat16)
    rotated = tl.where(d < 48, low, high)
    tl.store(output + index, rotated, mask=valid)


@triton.jit
def _rope_repeat_kv_kernel(key, value, cos, sin, key_out, value_out,
                           TOTAL_ELEMENTS: tl.constexpr, SEQ_LEN: tl.constexpr,
                           BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL_ELEMENTS
    d = index % 96
    rest = index // 96
    seq = rest % SEQ_LEN
    rest = rest // SEQ_LEN
    output_head = rest % 64
    batch = rest // 64
    source_head = output_head // 8

    half = d % 48
    source_base = ((batch * SEQ_LEN + seq) * 8 + source_head) * 96
    first = tl.load(key + source_base + half, mask=valid, other=0.0)
    second = tl.load(key + source_base + half + 48, mask=valid, other=0.0)
    trig_base = (batch * SEQ_LEN + seq) * 48 + half
    cos_value = tl.load(cos + trig_base, mask=valid, other=0.0)
    sin_value = tl.load(sin + trig_base, mask=valid, other=0.0)

    first_cos = (first * cos_value).to(tl.bfloat16)
    second_sin = (second * sin_value).to(tl.bfloat16)
    first_sin = (first * sin_value).to(tl.bfloat16)
    second_cos = (second * cos_value).to(tl.bfloat16)
    low = (first_cos - second_sin).to(tl.bfloat16)
    high = (first_sin + second_cos).to(tl.bfloat16)
    rotated = tl.where(d < 48, low, high)

    value_element = tl.load(value + source_base + d, mask=valid, other=0.0)
    tl.store(key_out + index, rotated, mask=valid)
    tl.store(value_out + index, value_element, mask=valid)


@triton.jit
def _swiglu_kernel(gate, up, output, TOTAL_ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL_ELEMENTS
    gate_value = tl.load(gate + index, mask=valid, other=0.0).to(tl.float32)
    up_value = tl.load(up + index, mask=valid, other=0.0)
    silu = (gate_value * tl.sigmoid(gate_value)).to(tl.bfloat16)
    tl.store(output + index, silu * up_value, mask=valid)


@triton.jit
def _scaled_masked_softmax_kernel(
    scores,
    mask,
    output,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < SEQ_LEN

    # scores is [B, 64, S, S], while mask is broadcast from [B, 1, S, S].
    batch = row // (64 * SEQ_LEN)
    query_pos = row % SEQ_LEN
    mask_row = (batch * SEQ_LEN + query_pos) * SEQ_LEN

    score = tl.load(scores + row * SEQ_LEN + col, mask=valid, other=-float("inf"))
    bias = tl.load(mask + mask_row + col, mask=valid, other=0.0)

    # These explicit bf16 casts reproduce the two eager bf16 roundings:
    # matmul_result * scale, followed by + attention_mask.
    scaled = (score.to(tl.float32) * 0.10206207261596575).to(tl.bfloat16)
    logits = (scaled + bias).to(tl.bfloat16).to(tl.float32)
    logits = tl.where(valid, logits, -float("inf"))

    logits = logits - tl.max(logits, axis=0)
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(output + row * SEQ_LEN + col, result, mask=valid)


def _scaled_masked_softmax(scores, attention_mask, seq_len):
    output = torch.empty_like(scores)
    block = triton.next_power_of_2(seq_len)
    rows = scores.numel() // seq_len
    _scaled_masked_softmax_kernel[(rows,)](
        scores,
        attention_mask,
        output,
        SEQ_LEN=seq_len,
        BLOCK=block,
        # CDNA wavefronts have 64 lanes; two waves give this reduction the
        # best occupancy across every sequence length in the workload set.
        num_warps=2,
        num_stages=1,
    )
    return output


def _rms_norm(x, weight, eps):
    output = torch.empty_like(x)
    rows = x.numel() // 6144
    _rms_norm_kernel[(rows,)](x, weight, output, eps, BLOCK=8192,
                              num_warps=8, num_stages=1)
    return output


def _add_rms_norm(x, update, weight, eps):
    residual = torch.empty_like(x)
    output = torch.empty_like(x)
    rows = x.numel() // 6144
    _add_rms_norm_kernel[(rows,)](
        x, update, weight, residual, output, eps, BLOCK=8192,
        num_warps=8, num_stages=1,
    )
    return residual, output


def _apply_rope_and_repeat(query, key, value, cos, sin, batch_size, seq_len):
    query_out = torch.empty((batch_size, 64, seq_len, 96),
                            device=query.device, dtype=query.dtype)
    key_out = torch.empty_like(query_out)
    value_out = torch.empty_like(query_out)
    query_elements = query.numel()
    block = 256
    # Passing the exact element count makes the final program's mask unambiguous.
    _rope_query_kernel[(triton.cdiv(query_elements, block),)](
        query, cos, sin, query_out, TOTAL_ELEMENTS=query_elements,
        SEQ_LEN=seq_len, BLOCK=block,
        num_warps=4, num_stages=1,
    )
    output_elements = query_out.numel()
    _rope_repeat_kv_kernel[(triton.cdiv(output_elements, block),)](
        key, value, cos, sin, key_out, value_out,
        TOTAL_ELEMENTS=output_elements, SEQ_LEN=seq_len, BLOCK=block,
        num_warps=4, num_stages=1,
    )
    return query_out, key_out, value_out


def _swiglu(gate, up):
    output = torch.empty_like(gate)
    elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(elements, 512),)](
        gate, up, output, TOTAL_ELEMENTS=elements, BLOCK=512,
        num_warps=1, num_stages=1,
    )
    return output


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    attention_mask,
    input_layernorm_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    post_attention_layernorm_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    rms_norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape

    residual = hidden_states
    normalized = _rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)

    query = F.linear(normalized, q_proj_weight)
    key = F.linear(normalized, k_proj_weight)
    value = F.linear(normalized, v_proj_weight)

    query, key, value = _apply_rope_and_repeat(
        query, key, value, cos, sin, batch_size, seq_len
    )

    scores = torch.matmul(query, key.transpose(2, 3))
    probabilities = _scaled_masked_softmax(scores, attention_mask, seq_len)
    attention = torch.matmul(probabilities, value)
    attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, 6144)

    attention = F.linear(attention, o_proj_weight)
    residual, normalized = _add_rms_norm(
        residual, attention, post_attention_layernorm_weight, rms_norm_eps
    )

    gate = F.linear(normalized, gate_proj_weight)
    up = F.linear(normalized, up_proj_weight)
    # Native kernels have lower launch cost for very small token counts; once
    # there is enough work, one-wave Triton avoids the intermediate SiLU tensor.
    if normalized.numel() >= 1024 * 6144:
        mlp_activation = _swiglu(gate, up)
    else:
        mlp_activation = F.silu(gate) * up
    hidden_states = F.linear(mlp_activation, down_proj_weight)
    return residual + hidden_states
