import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_bw_kernel(
    # pointers
    grad_probs_dropped_ptr,  # [B, H, S, S]
    attn_probs_ptr,          # [B, H, S, S]
    dropout_mask_ptr,        # [B, H, S, S]
    attn_mask_ptr,           # [B, S]
    grad_scores_ptr,         # [B, H, S, S]  output
    inv_dp,
    B, H, S,
    # strides
    stride_gb, stride_gh, stride_gq, stride_gk,
    BLOCK_S: tl.constexpr,
):
    # one program per (b, h, q)
    pid = tl.program_id(0)
    bh = pid // S
    q = pid % S
    b = bh // H
    h = bh % H

    offs = tl.arange(0, BLOCK_S)
    mask = offs < S

    row_offset = b * stride_gb + h * stride_gh + q * stride_gq
    gptr = grad_probs_dropped_ptr + row_offset
    aptr = attn_probs_ptr + row_offset
    dptr = dropout_mask_ptr + row_offset

    gapd = tl.load(gptr + offs * stride_gk, mask=mask, other=0.0)
    ap = tl.load(aptr + offs * stride_gk, mask=mask, other=0.0)
    dm = tl.load(dptr + offs * stride_gk, mask=mask, other=0.0)

    gap = gapd * dm * inv_dp
    sum_grad = tl.sum(gap * ap, axis=0)
    gas = ap * (gap - sum_grad)

    # masked_fill: where attn_mask[b,k]==0 -> 0
    am = tl.load(attn_mask_ptr + b * S + offs, mask=mask, other=0.0)
    gas = tl.where(am == 0, 0.0, gas)

    sptr = grad_scores_ptr + row_offset
    tl.store(sptr + offs * stride_gk, gas, mask=mask)


def fused_softmax_backward(grad_probs_dropped, attn_probs, dropout_mask, attn_mask, dropout_p):
    B, H, S, _ = grad_probs_dropped.shape
    grad_scores = torch.empty_like(grad_probs_dropped)
    inv_dp = 1.0 / (1.0 - dropout_p) if dropout_p > 0 else 1.0
    BLOCK_S = triton.next_power_of_2(S)
    grid = (B * H * S,)
    _softmax_bw_kernel[grid](
        grad_probs_dropped, attn_probs, dropout_mask, attn_mask, grad_scores,
        inv_dp, B, H, S,
        grad_probs_dropped.stride(0), grad_probs_dropped.stride(1),
        grad_probs_dropped.stride(2), grad_probs_dropped.stride(3),
        BLOCK_S=BLOCK_S,
    )
    return grad_scores


@torch.no_grad()
def run(
    grad_output, hidden_states, edit_region_mask,
    qkv_weight, qkv_bias, out_weight, out_bias,
    edit_region_bias, within_edit_bias, cross_edit_bias,
    attention_mask, query, key, value,
    attention_scores, attention_probs, attention_probs_dropped,
    attention_output, dropout_mask, scale, dropout_p,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 16
    head_dim = 64
    max_position_embeddings = 1024

    grad_attention_output = torch.matmul(grad_output, out_weight)
    grad_out_weight = torch.matmul(
        grad_output.reshape(-1, hidden_size).t(),
        attention_output.reshape(-1, hidden_size)
    )
    grad_out_bias = grad_output.sum(dim=[0, 1])

    grad_attention_output = grad_attention_output.reshape(
        batch_size, seq_len, num_attention_heads, head_dim
    )
    grad_attention_output = grad_attention_output.transpose(1, 2)

    grad_attention_probs_dropped = torch.matmul(
        grad_attention_output, value.transpose(-2, -1)
    )
    grad_value = torch.matmul(
        attention_probs_dropped.transpose(-2, -1), grad_attention_output
    )

    # FUSED softmax backward (replaces dropout grad + sum_grad + grad_scores + masked_fill)
    grad_attention_scores = fused_softmax_backward(
        grad_attention_probs_dropped, attention_probs, dropout_mask,
        attention_mask, dropout_p
    )

    edit_mask_q = edit_region_mask.unsqueeze(2)
    edit_mask_k = edit_region_mask.unsqueeze(1)
    cross_edit = (edit_mask_q * (1 - edit_mask_k) + (1 - edit_mask_q) * edit_mask_k).unsqueeze(1)
    grad_cross_edit_bias = (grad_attention_scores * cross_edit).sum(dim=[0, 2, 3], keepdim=True)
    grad_cross_edit_bias = grad_cross_edit_bias.squeeze(0).unsqueeze(-1).unsqueeze(-1)
    grad_cross_edit_bias = grad_cross_edit_bias.reshape(num_attention_heads, 1, 1)

    within_edit = (edit_mask_q * edit_mask_k).unsqueeze(1)
    grad_within_edit_bias = (grad_attention_scores * within_edit).sum(dim=[0, 2, 3], keepdim=True)
    grad_within_edit_bias = grad_within_edit_bias.squeeze(0).unsqueeze(-1).unsqueeze(-1)
    grad_within_edit_bias = grad_within_edit_bias.reshape(num_attention_heads, 1, 1)

    grad_edit_region_bias = torch.zeros_like(edit_region_bias)
    if seq_len <= max_position_embeddings:
        grad_edit_region_bias[:, :seq_len, :seq_len] = grad_attention_scores.sum(dim=0)

    grad_attention_scores_scaled = grad_attention_scores * scale
    grad_query = torch.matmul(grad_attention_scores_scaled, key)
    grad_key = torch.matmul(grad_attention_scores_scaled.transpose(-2, -1), query)

    grad_qkv = torch.stack([grad_query, grad_key, grad_value], dim=0)
    grad_qkv = grad_qkv.permute(1, 3, 0, 2, 4)
    grad_qkv = grad_qkv.reshape(batch_size, seq_len, 3 * hidden_size)

    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight)
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * hidden_size).t(),
        hidden_states.reshape(-1, hidden_size)
    )
    grad_qkv_bias = grad_qkv.sum(dim=[0, 1])

    return (
        grad_hidden_states, grad_qkv_weight, grad_qkv_bias,
        grad_out_weight, grad_out_bias, grad_edit_region_bias,
        grad_within_edit_bias, grad_cross_edit_bias,
    )
