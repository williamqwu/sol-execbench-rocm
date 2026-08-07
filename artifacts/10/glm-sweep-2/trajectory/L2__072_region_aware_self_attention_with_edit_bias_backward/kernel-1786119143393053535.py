import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _grad_scores_kernel(
    gap_ptr,            # [B,H,S,S]  grad_attention_probs (precomputed, bit-exact)
    ap_ptr,             # [B,H,S,S]  attention_probs
    am_ptr,             # [B,S]      attention_mask
    sg_ptr,             # [B,H,S]    sum_grad (precomputed, bit-exact)
    gs_ptr,             # [B,H,S,S]  output grad_attention_scores
    B, H, S,
    s0, s1, s2, s3,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // S
    q = pid % S
    b = bh // H
    h = bh % H
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    ro = b * s0 + h * s1 + q * s2
    gap = tl.load(gap_ptr + ro + offs * s3, mask=mask, other=0.0)
    ap = tl.load(ap_ptr + ro + offs * s3, mask=mask, other=0.0)
    sg = tl.load(sg_ptr + b * H * S + h * S + q)
    gas = ap * (gap - sg)
    am = tl.load(am_ptr + b * S + offs, mask=mask, other=0.0)
    gas = tl.where(am == 0, 0.0, gas)
    tl.store(gs_ptr + ro + offs * s3, gas, mask=mask)


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

    # dropout grad (bit-exact eager)
    if dropout_p > 0:
        grad_attention_probs = grad_attention_probs_dropped * dropout_mask / (1 - dropout_p)
    else:
        grad_attention_probs = grad_attention_probs_dropped

    # softmax grad: sum_grad (bit-exact eager)
    sum_grad = (grad_attention_probs * attention_probs).sum(dim=-1, keepdim=True)

    # FUSED: grad_scores = ap*(gap - sum_grad) + masked_fill  (Triton, bit-exact)
    grad_attention_scores = torch.empty_like(attention_probs)
    BLOCK_S = triton.next_power_of_2(seq_len)
    grid = (batch_size * num_attention_heads * seq_len,)
    _grad_scores_kernel[grid](
        grad_attention_probs, attention_probs, attention_mask,
        sum_grad.view(batch_size, num_attention_heads, seq_len),
        grad_attention_scores,
        batch_size, num_attention_heads, seq_len,
        grad_attention_probs.stride(0), grad_attention_probs.stride(1),
        grad_attention_probs.stride(2), grad_attention_probs.stride(3),
        BLOCK_S=BLOCK_S,
    )

    # bias grads (bit-exact eager)
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
