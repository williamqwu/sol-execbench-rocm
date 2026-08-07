import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(gate, up, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask).to(tl.float32)
    value = gate_value * tl.sigmoid(gate_value) * up_value
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _silu_mul_strided_kernel(
    gate,
    up,
    output,
    gate_row_stride,
    up_row_stride,
    output_row_stride,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    mask = cols < N_COLS
    gate_value = tl.load(gate + row * gate_row_stride + cols, mask=mask).to(tl.float32)
    up_value = tl.load(up + row * up_row_stride + cols, mask=mask).to(tl.float32)
    value = gate_value * tl.sigmoid(gate_value) * up_value
    tl.store(output + row * output_row_stride + cols, value, mask=mask)


@triton.jit
def _cat2_kernel(first, second, output, first_size, total_size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first_mask = offsets < first_size
    second_offsets = offsets - first_size
    first_value = tl.load(first + offsets, mask=first_mask, other=0.0)
    second_value = tl.load(
        second + second_offsets,
        mask=(~first_mask) & (offsets < total_size),
        other=0.0,
    )
    tl.store(output + offsets, first_value + second_value, mask=offsets < total_size)


@triton.jit
def _cat3_kernel(
    first, second, third, output, first_size, second_end, total_size, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first_mask = offsets < first_size
    second_mask = (offsets >= first_size) & (offsets < second_end)
    third_mask = (offsets >= second_end) & (offsets < total_size)
    first_value = tl.load(first + offsets, mask=first_mask, other=0.0)
    second_value = tl.load(second + offsets - first_size, mask=second_mask, other=0.0)
    third_value = tl.load(third + offsets - second_end, mask=third_mask, other=0.0)
    tl.store(
        output + offsets,
        first_value + second_value + third_value,
        mask=offsets < total_size,
    )


@triton.jit
def _add_rms_kernel(
    input,
    residual,
    weight,
    residual_out,
    norm_out,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    input_value = tl.load(input + row * BLOCK + cols).to(tl.float32)
    residual_value = tl.load(residual + row * BLOCK + cols).to(tl.float32)
    summed = (input_value + residual_value).to(tl.bfloat16)
    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / BLOCK
    weight_value = tl.load(weight + cols).to(tl.float32)
    normed = summed_fp32 * tl.rsqrt(variance + eps) * weight_value
    tl.store(residual_out + row * BLOCK + cols, summed)
    tl.store(norm_out + row * BLOCK + cols, normed)


def _rms_norm(x, weight, eps):
    # PyTorch's fused kernel uses fp32 for the reduction and returns bf16,
    # matching the explicit conversion boundaries in the specification.
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _apply_rope(q, k, batch_size, seq_len, rope_theta):
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, 128, 2, dtype=torch.float32, device=q.device) / 128)
    )
    freqs = torch.outer(torch.arange(seq_len, device=q.device).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype).view(1, 1, seq_len, 128)
    sin = emb.sin().to(q.dtype).view(1, 1, seq_len, 128)
    q_rot = torch.cat((-q[..., 64:], q[..., :64]), dim=-1)
    k_rot = torch.cat((-k[..., 64:], k[..., :64]), dim=-1)
    return q * cos + q_rot * sin, k * cos + k_rot * sin


_apply_rope_fused = torch.compile(_apply_rope, fullgraph=True, dynamic=True)


def _rope_tables(q, rope_theta):
    seq_len = q.shape[2]
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, 128, 2, dtype=torch.float32, device=q.device) / 128)
    )
    freqs = torch.outer(torch.arange(seq_len, device=q.device).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        emb.cos().to(q.dtype).view(1, 1, seq_len, 128),
        emb.sin().to(q.dtype).view(1, 1, seq_len, 128),
    )


def _rotate_with_tables(q, k, cos, sin):
    q_rot = torch.cat((-q[..., 64:], q[..., :64]), dim=-1)
    k_rot = torch.cat((-k[..., 64:], k[..., :64]), dim=-1)
    return q * cos + q_rot * sin, k * cos + k_rot * sin


_rope_tables = torch.compile(_rope_tables, fullgraph=True, dynamic=True)
_rotate_with_tables = torch.compile(
    _rotate_with_tables, fullgraph=True, dynamic=True
)


def _apply_rope(q, k, batch_size, seq_len, rope_theta):
    if batch_size * seq_len >= 3072:
        cos, sin = _rope_tables(q, rope_theta)
        return _rotate_with_tables(q, k, cos, sin)
    return _apply_rope_fused(q, k, batch_size, seq_len, rope_theta)


def _silu_mul(gate, up):
    rows = gate.numel() // 8192
    output = torch.empty_like(gate, memory_format=torch.contiguous_format)
    if gate.is_contiguous():
        if rows <= 1024:
            block, num_warps = 1024, 4
        else:
            block, num_warps = 4096, 8
        _silu_mul_kernel[(triton.cdiv(gate.numel(), block),)](
            gate,
            up,
            output,
            gate.numel(),
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        if rows < 512:
            block, num_warps = 512, 4
        else:
            block, num_warps = 1024, 8
        _silu_mul_strided_kernel[(triton.cdiv(8192, block), rows)](
            gate,
            up,
            output,
            gate.stride(-2),
            up.stride(-2),
            output.stride(-2),
            N_COLS=8192,
            BLOCK=block,
            num_warps=num_warps,
        )
    return output


def _add_rms(input, residual, weight, eps):
    rows = input.numel() // input.shape[-1]
    residual_out = torch.empty_like(input)
    norm_out = torch.empty_like(input)
    _add_rms_kernel[(rows,)](
        input,
        residual,
        weight,
        residual_out,
        norm_out,
        eps,
        BLOCK=2048,
        num_warps=4,
    )
    return residual_out, norm_out


def _cat2_weights(first, second):
    output = torch.empty(
        (first.shape[0] + second.shape[0], first.shape[1]),
        dtype=first.dtype,
        device=first.device,
    )
    total = output.numel()
    if total > 10_000_000:
        block, num_warps = 4096, 8
    else:
        block, num_warps = 256, 4
    _cat2_kernel[(triton.cdiv(total, block),)](
        first,
        second,
        output,
        first.numel(),
        total,
        BLOCK=block,
        num_warps=num_warps,
    )
    return output


@torch.no_grad()
def run(
    hidden_states,
    encoder_hidden_states,
    self_attn_norm_weight,
    self_attn_q_weight,
    self_attn_k_weight,
    self_attn_v_weight,
    self_attn_o_weight,
    cross_attn_norm_weight,
    cross_attn_q_weight,
    cross_attn_k_weight,
    cross_attn_v_weight,
    cross_attn_o_weight,
    mlp_norm_weight,
    mlp_gate_weight,
    mlp_up_weight,
    mlp_down_weight,
    norm_eps,
    rope_theta,
):
    batch_size, seq_len, _ = hidden_states.shape
    encoder_seq_len = encoder_hidden_states.shape[1]

    residual = hidden_states
    x = _rms_norm(hidden_states, self_attn_norm_weight, norm_eps)
    qkv = F.linear(
        x,
        torch.cat(
            (self_attn_q_weight, self_attn_k_weight, self_attn_v_weight), dim=0
        ),
    )
    q = qkv[..., :2048].view(batch_size, seq_len, 16, 128).transpose(1, 2)
    k = qkv[..., 2048:2560].view(batch_size, seq_len, 4, 128).transpose(1, 2)
    v = qkv[..., 2560:].view(batch_size, seq_len, 4, 128).transpose(1, 2)
    q, k = _apply_rope(q, k, batch_size, seq_len, rope_theta)
    x = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True
    ).transpose(1, 2).contiguous().view(batch_size, seq_len, 2048)
    residual, x = _add_rms(
        F.linear(x, self_attn_o_weight),
        residual,
        cross_attn_norm_weight,
        norm_eps,
    )
    q = F.linear(x, cross_attn_q_weight).view(batch_size, seq_len, 16, 128).transpose(1, 2)
    encoder_rows = batch_size * encoder_seq_len
    if encoder_rows not in (8192, 16384):
        kv = F.linear(
            encoder_hidden_states,
            _cat2_weights(cross_attn_k_weight, cross_attn_v_weight),
        )
        k, v = kv.chunk(2, dim=-1)
    else:
        k = F.linear(encoder_hidden_states, cross_attn_k_weight)
        v = F.linear(encoder_hidden_states, cross_attn_v_weight)
    k = k.view(batch_size, encoder_seq_len, 16, 128).transpose(1, 2)
    v = v.view(batch_size, encoder_seq_len, 16, 128).transpose(1, 2)
    x = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False
    ).transpose(1, 2).contiguous().view(batch_size, seq_len, 2048)
    residual, x = _add_rms(
        F.linear(x, cross_attn_o_weight), residual, mlp_norm_weight, norm_eps
    )
    if batch_size * seq_len <= 512:
        gate_up = F.linear(x, _cat2_weights(mlp_gate_weight, mlp_up_weight))
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = F.linear(x, mlp_gate_weight)
        up = F.linear(x, mlp_up_weight)
    x = F.linear(_silu_mul(gate, up), mlp_down_weight)
    return residual + x
