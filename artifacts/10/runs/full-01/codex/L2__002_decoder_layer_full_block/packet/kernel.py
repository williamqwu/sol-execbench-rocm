import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scale_mask_kernel(
    scores, mask, score_elems, USE_I64: tl.constexpr, BLOCK: tl.constexpr
):
    bh = tl.program_id(1)
    first = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    for base in tl.range(first, score_elems, step):
        offsets = base + tl.arange(0, BLOCK)
        valid = offsets < score_elems
        if USE_I64:
            score_offsets = bh.to(tl.int64) * score_elems + offsets
        else:
            score_offsets = bh * score_elems + offsets
        mask_offsets = (bh // 32) * score_elems + offsets
        values = tl.load(scores + score_offsets, mask=valid)
        mask_values = tl.load(mask + mask_offsets, mask=valid)
        scaled = values * 0.08838834764831845
        result = scaled + mask_values
        tl.store(scores + score_offsets, result, mask=valid)


@triton.jit
def _rope_qk_kernel(q, k, cos, sin, q_pairs, k_pairs, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q_valid = offsets < q_pairs
    q_head_token = offsets // 64
    q_dim = offsets % 64
    q_token = q_head_token // 32
    q_lo_offset = q_head_token * 128 + q_dim
    q_hi_offset = q_lo_offset + 64
    q_lo = tl.load(q + q_lo_offset, mask=q_valid)
    q_hi = tl.load(q + q_hi_offset, mask=q_valid)
    q_cos_lo = tl.load(cos + q_token * 128 + q_dim, mask=q_valid)
    q_cos_hi = tl.load(cos + q_token * 128 + q_dim + 64, mask=q_valid)
    q_sin_lo = tl.load(sin + q_token * 128 + q_dim, mask=q_valid)
    q_sin_hi = tl.load(sin + q_token * 128 + q_dim + 64, mask=q_valid)
    q_a = q_lo * q_cos_lo
    q_b = (-q_hi) * q_sin_lo
    q_c = q_hi * q_cos_hi
    q_d = q_lo * q_sin_hi
    tl.store(q + q_lo_offset, q_a + q_b, mask=q_valid)
    tl.store(q + q_hi_offset, q_c + q_d, mask=q_valid)

    k_valid = offsets < k_pairs
    k_head_token = offsets // 64
    k_dim = offsets % 64
    k_token = k_head_token // 8
    k_lo_offset = k_head_token * 128 + k_dim
    k_hi_offset = k_lo_offset + 64
    k_lo = tl.load(k + k_lo_offset, mask=k_valid)
    k_hi = tl.load(k + k_hi_offset, mask=k_valid)
    k_cos_lo = tl.load(cos + k_token * 128 + k_dim, mask=k_valid)
    k_cos_hi = tl.load(cos + k_token * 128 + k_dim + 64, mask=k_valid)
    k_sin_lo = tl.load(sin + k_token * 128 + k_dim, mask=k_valid)
    k_sin_hi = tl.load(sin + k_token * 128 + k_dim + 64, mask=k_valid)
    k_a = k_lo * k_cos_lo
    k_b = (-k_hi) * k_sin_lo
    k_c = k_hi * k_cos_hi
    k_d = k_lo * k_sin_hi
    tl.store(k + k_lo_offset, k_a + k_b, mask=k_valid)
    tl.store(k + k_hi_offset, k_c + k_d, mask=k_valid)


@triton.jit
def _norm_scale_kernel(x, inv_rms, weight, out, elem_count, BLOCK: tl.constexpr):
    first = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    for base in tl.range(first, elem_count, step):
        offsets = base + tl.arange(0, BLOCK)
        valid = offsets < elem_count
        values = tl.load(x + offsets, mask=valid)
        scales = tl.load(inv_rms + offsets // 4096, mask=valid)
        weights = tl.load(weight + offsets % 4096, mask=valid)
        normalized = values * scales
        result = weights * normalized
        tl.store(out + offsets, result, mask=valid)


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
    token_count = batch_size * seq_len
    residual = hidden_states

    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + rms_norm_eps)
    if token_count >= 1024:
        hidden_states = torch.empty_like(hidden_states)
        elem_count = token_count * 4096
        norm_programs = min(triton.cdiv(elem_count, 1024), 1024)
        _norm_scale_kernel[(norm_programs,)](
            x,
            inv_rms,
            input_layernorm_weight,
            hidden_states,
            elem_count,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        x = x * inv_rms
        hidden_states = input_layernorm_weight * x.to(hidden_states.dtype)

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    q_pairs = token_count * 32 * 64
    k_pairs = token_count * 8 * 64
    rope_block = 256 if token_count < 1024 else 1024
    _rope_qk_kernel[(triton.cdiv(q_pairs, rope_block),)](
        query_states,
        key_states,
        cos,
        sin,
        q_pairs,
        k_pairs,
        BLOCK=rope_block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    query_states = query_states.view(batch_size, seq_len, 32, 128).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, 8, 4, seq_len, 128
    ).reshape(batch_size, 32, seq_len, 128)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, 8, 4, seq_len, 128
    ).reshape(batch_size, 32, seq_len, 128)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    score_elems = seq_len * seq_len
    score_programs = min(triton.cdiv(score_elems, 2048), 512)
    use_i64 = batch_size * 32 * score_elems > 2147483648
    _scale_mask_kernel[(score_programs, batch_size * 32)](
        attn_weights,
        attention_mask,
        score_elems,
        USE_I64=use_i64,
        BLOCK=2048,
        num_warps=4,
        enable_fp_fusion=False,
    )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, 4096)

    hidden_states = torch.addmm(
        residual.view(-1, 4096),
        attn_output.view(-1, 4096),
        o_proj_weight.t(),
    ).view(batch_size, seq_len, 4096)
    residual = hidden_states

    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + rms_norm_eps)
    if token_count >= 1024:
        hidden_states = torch.empty_like(hidden_states)
        _norm_scale_kernel[(norm_programs,)](
            x,
            inv_rms,
            post_attention_layernorm_weight,
            hidden_states,
            elem_count,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        x = x * inv_rms
        hidden_states = post_attention_layernorm_weight * x.to(hidden_states.dtype)

    gate = F.linear(hidden_states, gate_proj_weight)
    F.silu(gate, inplace=True)
    up = F.linear(hidden_states, up_proj_weight)
    gate.mul_(up)
    return torch.addmm(
        residual.view(-1, 4096),
        gate.view(-1, 14336),
        down_proj_weight.t(),
    ).view(batch_size, seq_len, 4096)
