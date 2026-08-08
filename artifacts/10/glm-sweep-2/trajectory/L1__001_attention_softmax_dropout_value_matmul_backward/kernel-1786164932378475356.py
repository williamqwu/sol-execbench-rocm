import torch
import triton
import triton.language as tl

torch.set_float32_matmul_precision("high")

_compiled_run = None


@triton.jit
def _softmax_dropout_backward_kernel(
    grad_attn_scores_ptr,
    grad_aw_dropped_ptr,
    attn_weights_ptr,
    attn_weights_dropped_ptr,
    seq_len_kv,
    sb, sh, sq, skv,
    BLOCK_KV: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_qg = tl.program_id(2)
    kv_offsets = tl.arange(0, BLOCK_KV)
    col_mask = kv_offsets < seq_len_kv
    for r in range(ROWS_PER_PROGRAM):
        pid_q = pid_qg * ROWS_PER_PROGRAM + r
        row_offset = pid_b * sb + pid_h * sh + pid_q * sq
        gaw = tl.load(grad_aw_dropped_ptr + row_offset + kv_offsets * skv, mask=col_mask, other=0.0)
        aw = tl.load(attn_weights_ptr + row_offset + kv_offsets * skv, mask=col_mask, other=0.0).to(tl.float32)
        awd = tl.load(attn_weights_dropped_ptr + row_offset + kv_offsets * skv, mask=col_mask, other=0.0).to(tl.float32)
        sum_term = tl.sum(gaw * awd)
        grad_scores = gaw * awd - aw * sum_term
        tl.store(grad_attn_scores_ptr + row_offset + kv_offsets * skv, grad_scores.to(tl.bfloat16), mask=col_mask)


def _softmax_dropout_backward(grad_aw_dropped, attn_weights, attn_weights_dropped, attention_dropout):
    B, H, S_q, S_kv = grad_aw_dropped.shape
    grad_attn_scores = torch.empty_like(attn_weights)
    BLOCK_KV = max(triton.next_power_of_2(S_kv), 16)
    gaw_c = grad_aw_dropped.contiguous()
    aw_c = attn_weights.contiguous()
    awd_c = attn_weights_dropped.contiguous()
    rows_per_program = 4 if S_q >= 256 else 1
    grid_q = (S_q + rows_per_program - 1) // rows_per_program
    grid = (B, H, grid_q)
    num_warps = 8 if S_kv >= 512 else (4 if S_kv >= 128 else 1)
    _softmax_dropout_backward_kernel[grid](
        grad_attn_scores, gaw_c, aw_c, awd_c,
        S_kv,
        gaw_c.stride(0), gaw_c.stride(1), gaw_c.stride(2), gaw_c.stride(3),
        BLOCK_KV=BLOCK_KV, ROWS_PER_PROGRAM=rows_per_program, num_warps=num_warps,
    )
    return grad_attn_scores


def _run_impl(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    value_states_expanded = value_states
    if num_key_value_groups > 1:
        value_states_expanded = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len_kv, head_dim)

    grad_attn_output_transposed = grad_attn_output.transpose(1, 2).to(torch.float32)

    grad_attn_weights_dropped = torch.matmul(
        grad_attn_output_transposed,
        value_states_expanded.to(torch.float32).transpose(-2, -1)
    )

    grad_attn_scores = _softmax_dropout_backward(
        grad_attn_weights_dropped, attn_weights, attn_weights_dropped, attention_dropout
    )

    grad_value_states_expanded = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1),
        grad_attn_output_transposed
    )

    if num_key_value_groups > 1:
        grad_value_states_expanded = grad_value_states_expanded.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        )
        grad_value_states = grad_value_states_expanded.sum(dim=2)
    else:
        grad_value_states = grad_value_states_expanded

    grad_value_states = grad_value_states.to(torch.bfloat16)

    return grad_attn_scores, grad_value_states


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    global _compiled_run
    if _compiled_run is None:
        _compiled_run = torch.compile(_run_impl, dynamic=True)
    return _compiled_run(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
