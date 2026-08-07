import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rope_qk_token_kernel(
    q_ptr,
    k_ptr,
    pos0_ptr,
    pos1_ptr,
    pos2_ptr,
    cos0_ptr,
    sin0_ptr,
    cos1_ptr,
    sin1_ptr,
    cos2_ptr,
    sin2_ptr,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    axis = tl.program_id(1)
    pair = tl.arange(0, BLOCK)
    mask = pair < 24 * 21
    dim = pair % 21
    head = pair // 21
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN
    if axis == 0:
        pos = tl.load(pos0_ptr + seq)
        cos = tl.load(cos0_ptr + pos * 42 + dim, mask=mask, other=0.0)
        sin = tl.load(sin0_ptr + pos * 42 + dim, mask=mask, other=0.0)
    elif axis == 1:
        pos = tl.load(pos1_ptr + seq)
        cos = tl.load(cos1_ptr + pos * 42 + dim, mask=mask, other=0.0)
        sin = tl.load(sin1_ptr + pos * 42 + dim, mask=mask, other=0.0)
    else:
        pos = tl.load(pos2_ptr + seq)
        cos = tl.load(cos2_ptr + pos * 42 + dim, mask=mask, other=0.0)
        sin = tl.load(sin2_ptr + pos * 42 + dim, mask=mask, other=0.0)
    base = (((token * 24 + head) * 128) + axis * 42 + dim)
    q0 = tl.load(q_ptr + base, mask=mask)
    q1 = tl.load(q_ptr + base + 21, mask=mask)
    k0 = tl.load(k_ptr + base, mask=mask)
    k1 = tl.load(k_ptr + base + 21, mask=mask)
    q0_cos = q0 * cos
    q1_sin = q1 * sin
    q0_sin = q0 * sin
    q1_cos = q1 * cos
    k0_cos = k0 * cos
    k1_sin = k1 * sin
    k0_sin = k0 * sin
    k1_cos = k1 * cos
    tl.store(q_ptr + base, q0_cos - q1_sin, mask=mask)
    tl.store(q_ptr + base + 21, q0_sin + q1_cos, mask=mask)
    tl.store(k_ptr + base, k0_cos - k1_sin, mask=mask)
    tl.store(k_ptr + base + 21, k0_sin + k1_cos, mask=mask)


@triton.jit
def _modulate_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    SEQ_LEN: tl.constexpr,
    MOD_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = dim < 3072
    offsets = token * 3072 + dim
    batch = token // SEQ_LEN
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + batch * MOD_STRIDE + dim, mask=mask)
    shift = tl.load(shift_ptr + batch * MOD_STRIDE + dim, mask=mask)
    scale_one = 1.0 + scale
    product = x * scale_one
    tl.store(x_ptr + offsets, product + shift, mask=mask)


@triton.jit
def _gated_residual_kernel(
    attn_ptr,
    residual_ptr,
    gate_ptr,
    SEQ_LEN: tl.constexpr,
    MOD_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = dim < 3072
    offsets = token * 3072 + dim
    batch = token // SEQ_LEN
    attn = tl.load(attn_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + batch * MOD_STRIDE + dim, mask=mask)
    gated = gate * attn
    tl.store(attn_ptr + offsets, residual + gated, mask=mask)


@triton.jit
def _concat_context_kv_kernel(
    k_image_ptr,
    k_context_ptr,
    v_image_ptr,
    v_context_ptr,
    k_out_ptr,
    v_out_ptr,
    IMAGE_LEN: tl.constexpr,
    CONTEXT_LEN: tl.constexpr,
    TOTAL_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = dim < 3072
    offsets = row * 3072 + dim
    pos = row % TOTAL_LEN
    batch = row // TOTAL_LEN
    if pos < IMAGE_LEN:
        input_offset = (batch * IMAGE_LEN + pos) * 3072 + dim
        k_value = tl.load(k_image_ptr + input_offset, mask=valid)
        v_value = tl.load(v_image_ptr + input_offset, mask=valid)
    else:
        input_offset = (batch * CONTEXT_LEN + pos - IMAGE_LEN) * 3072 + dim
        k_value = tl.load(k_context_ptr + input_offset, mask=valid)
        v_value = tl.load(v_context_ptr + input_offset, mask=valid)
    tl.store(k_out_ptr + offsets, k_value, mask=valid)
    tl.store(v_out_ptr + offsets, v_value, mask=valid)


@torch.no_grad()
def run(
    hidden_states,
    timestep_embedding,
    encoder_hidden_states,
    adaln_linear_weight,
    adaln_linear_bias,
    to_q_weight,
    to_q_bias,
    to_k_weight,
    to_k_bias,
    to_v_weight,
    to_v_bias,
    to_k_context_weight,
    to_k_context_bias,
    to_v_context_weight,
    to_v_context_bias,
    to_out_weight,
    to_out_bias,
    pos_idx_axis0,
    pos_idx_axis1,
    pos_idx_axis2,
    rope_cos_axis0,
    rope_sin_axis0,
    rope_cos_axis1,
    rope_sin_axis1,
    rope_cos_axis2,
    rope_sin_axis2,
    is_joint_block,
):
    batch, seq_len, hidden_size = hidden_states.shape
    num_heads = 24
    head_dim = 128
    residual = hidden_states

    timestep_activated = torch.sigmoid(timestep_embedding)
    torch.mul(timestep_embedding, timestep_activated, out=timestep_activated)
    if batch >= 32:
        modulation_width = 9216
    elif batch == 1:
        modulation_width = 16385
    else:
        modulation_width = 16416
    modulation = F.linear(
        timestep_activated,
        adaln_linear_weight[:modulation_width],
        adaln_linear_bias[:modulation_width],
    )
    scale_msa = modulation[:, :hidden_size]
    shift_msa = modulation[:, hidden_size : 2 * hidden_size]
    gate_msa = modulation[:, 2 * hidden_size : 3 * hidden_size]

    hidden_states_normalized = F.layer_norm(hidden_states, (hidden_size,))
    token_count = batch * seq_len
    if token_count <= 512:
        affine_block, affine_warps = 128, 2
    elif token_count <= 1024:
        affine_block, affine_warps = 512, 8
    else:
        affine_block, affine_warps = 1024, 8
    _modulate_kernel[(token_count, triton.cdiv(hidden_size, affine_block))](
        hidden_states_normalized,
        scale_msa,
        shift_msa,
        SEQ_LEN=seq_len,
        MOD_STRIDE=modulation_width,
        BLOCK=affine_block,
        num_warps=affine_warps,
        enable_fp_fusion=False,
    )
    hidden_states_modulated = hidden_states_normalized

    q = F.linear(hidden_states_modulated, to_q_weight, to_q_bias)
    k = F.linear(hidden_states_modulated, to_k_weight, to_k_bias)
    v = F.linear(hidden_states_modulated, to_v_weight, to_v_bias)
    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_heads, head_dim)
    v = v.view(batch, seq_len, num_heads, head_dim)

    _rope_qk_token_kernel[(token_count, 3)](
        q,
        k,
        pos_idx_axis0,
        pos_idx_axis1,
        pos_idx_axis2,
        rope_cos_axis0,
        rope_sin_axis0,
        rope_cos_axis1,
        rope_sin_axis1,
        rope_cos_axis2,
        rope_sin_axis2,
        SEQ_LEN=seq_len,
        BLOCK=512,
        num_warps=8,
        enable_fp_fusion=False,
    )
    if is_joint_block == 1:
        text_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states_normalized = F.layer_norm(
            encoder_hidden_states, (4096,)
        )
        k_context = F.linear(
            encoder_hidden_states_normalized,
            to_k_context_weight,
            to_k_context_bias,
        ).view(batch, text_seq_len, num_heads, head_dim)
        v_context = F.linear(
            encoder_hidden_states_normalized,
            to_v_context_weight,
            to_v_context_bias,
        ).view(batch, text_seq_len, num_heads, head_dim)
        total_len = seq_len + text_seq_len
        k_joined = torch.empty(
            (batch, total_len, num_heads, head_dim),
            device=k.device,
            dtype=k.dtype,
        )
        v_joined = torch.empty_like(k_joined)
        if batch * total_len <= 320:
            concat_block, concat_warps = 128, 2
        else:
            concat_block, concat_warps = 1024, 8
        _concat_context_kv_kernel[
            (batch * total_len, triton.cdiv(hidden_size, concat_block))
        ](
            k,
            k_context,
            v,
            v_context,
            k_joined,
            v_joined,
            IMAGE_LEN=seq_len,
            CONTEXT_LEN=text_seq_len,
            TOTAL_LEN=total_len,
            BLOCK=concat_block,
            num_warps=concat_warps,
        )
        k = k_joined
        v = v_joined

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    attn_scores.mul_(1.0 / math.sqrt(128))
    attn_probs = torch.softmax(attn_scores, dim=-1, out=attn_scores)
    attn_output = torch.matmul(attn_probs, v)
    attn_output = (
        attn_output.transpose(1, 2)
        .contiguous()
        .view(batch, seq_len, hidden_size)
    )
    attn_output = F.linear(attn_output, to_out_weight, to_out_bias)
    _gated_residual_kernel[(token_count, triton.cdiv(hidden_size, affine_block))](
        attn_output,
        residual,
        gate_msa,
        SEQ_LEN=seq_len,
        MOD_STRIDE=modulation_width,
        BLOCK=affine_block,
        num_warps=affine_warps,
        enable_fp_fusion=False,
    )
    return attn_output
