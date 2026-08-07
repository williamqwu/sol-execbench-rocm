import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_GROUPED_BOTH = frozenset(
    {
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (2, 1571),
        (8, 256),
        (8, 373),
        (32, 128),
        (64, 128),
    }
)
_GROUPED_QK = _GROUPED_BOTH | {(2, 256)}
_GROUPED_AV = _GROUPED_BOTH | {(32, 256)}


@triton.jit
def _scale_causal_kernel(
    scores,
    seq_len: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    tile_row = tl.program_id(0)
    tile_col = tl.program_id(1)
    batch_head = tl.program_id(2)
    rows = tile_row * block_rows + tl.arange(0, block_rows)[:, None]
    cols = tile_col * block_cols + tl.arange(0, block_cols)[None, :]
    mask = (rows < seq_len) & (cols < seq_len)
    offsets = (batch_head * seq_len + rows) * seq_len + cols
    values = tl.load(scores + offsets, mask=mask)
    values = values * 0.08838834764831845
    values = tl.where(cols > rows, float("-inf"), values)
    tl.store(scores + offsets, values, mask=mask)


def _scale_and_mask(scores, seq_len):
    block_cols = 512 if seq_len >= 8192 else min(triton.next_power_of_2(seq_len), 256)
    block_rows = 1024 // block_cols
    grid = (
        triton.cdiv(seq_len, block_rows),
        triton.cdiv(seq_len, block_cols),
        scores.numel() // (seq_len * seq_len),
    )
    _scale_causal_kernel[grid](
        scores,
        seq_len=seq_len,
        block_rows=block_rows,
        block_cols=block_cols,
        num_warps=4,
    )


@torch.no_grad()
def run(
    hidden_states,
    input_layernorm_weight,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    o_proj_weight,
    rope_cos,
    rope_sin,
    post_attention_layernorm_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    rms_norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape

    residual = hidden_states
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    variance.add_(rms_norm_eps).rsqrt_()
    hidden_states = hidden_fp32 * variance
    hidden_states.mul_(input_layernorm_weight)

    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    query_states = query_states.view(batch_size, seq_len, 28, 128).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, 4, 128).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 4, 128).transpose(1, 2)

    cos_combined = torch.cat(
        (rope_cos[0, ..., :32], rope_cos[1, ..., 32:80], rope_cos[2, ..., 80:128]),
        dim=-1,
    )[:, :seq_len, :].unsqueeze(1)
    sin_combined = torch.cat(
        (rope_sin[0, ..., :32], rope_sin[1, ..., 32:80], rope_sin[2, ..., 80:128]),
        dim=-1,
    )[:, :seq_len, :].unsqueeze(1)

    query_states = query_states * cos_combined + torch.cat(
        (-query_states[..., 64:], query_states[..., :64]), dim=-1
    ) * sin_combined
    key_states = key_states * cos_combined + torch.cat(
        (-key_states[..., 64:], key_states[..., :64]), dim=-1
    ) * sin_combined

    shape = (batch_size, seq_len)
    grouped_qk = shape in _GROUPED_QK
    grouped_av = shape in _GROUPED_AV
    if grouped_qk:
        grouped_query = query_states.view(batch_size, 4, 7, seq_len, 128)
        attn_weights = torch.einsum(
            "bgqsd,bgtd->bgqst", grouped_query, key_states
        )
    else:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, 4, 7, seq_len, 128
        ).reshape(batch_size, 28, seq_len, 128)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

    _scale_and_mask(attn_weights, seq_len)
    torch.softmax(attn_weights, dim=-1, dtype=torch.float32, out=attn_weights)

    if grouped_av:
        if not grouped_qk:
            attn_weights = attn_weights.view(
                batch_size, 4, 7, seq_len, seq_len
            )
        attn_output = torch.einsum(
            "bgqst,bgtd->bgqsd", attn_weights, value_states
        ).view(batch_size, 28, seq_len, 128)
    else:
        if grouped_qk:
            attn_weights = attn_weights.view(
                batch_size, 28, seq_len, seq_len
            )
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, 4, 7, seq_len, 128
        ).reshape(batch_size, 28, seq_len, 128)
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        batch_size, seq_len, 3584
    )
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = attn_output.add_(residual)

    residual = hidden_states
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    variance.add_(rms_norm_eps).rsqrt_()
    hidden_states = hidden_fp32 * variance
    hidden_states.mul_(post_attention_layernorm_weight)

    gate_output = F.linear(hidden_states, gate_proj_weight)
    up_output = F.linear(hidden_states, up_proj_weight)
    F.silu(gate_output, inplace=True)
    gate_output.mul_(up_output)
    hidden_states = F.linear(gate_output, down_proj_weight)
    return hidden_states.add_(residual)
