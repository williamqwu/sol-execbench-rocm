import torch
import math


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128

    device = q.device
    inv_log2 = 1.0 / math.log(2.0)

    # Single batch: varlen with known max_q = total_q, direct LSE slice
    if len_indptr == 2:
        result = torch.ops.aten._flash_attention_forward(
            q, k, v, qo_indptr, kv_indptr, total_q, total_q,
            0.0, True, False, scale=sm_scale
        )
        output = result[0]
        lse = result[1][0, :, :total_q].transpose(0, 1) * inv_log2
        return output, lse

    # Multi-batch: use varlen flash attention (single kernel launch)
    # For small workloads, use total_q as max_q to avoid GPU sync
    # For large workloads, compute exact max_q (sync is worth it)
    if total_q < 1000:
        max_q = total_q
    else:
        max_q = (qo_indptr[1:] - qo_indptr[:-1]).max().item()

    result = torch.ops.aten._flash_attention_forward(
        q, k, v, qo_indptr, kv_indptr, max_q, max_q,
        0.0, True, False, scale=sm_scale
    )
    output = result[0]
    lse_packed = result[1]  # [batch, 32, max_q]

    # Unpack LSE: [batch, 32, max_q] -> [total_q, 32]
    positions = torch.arange(total_q, device=device)
    batch_idx = torch.searchsorted(qo_indptr, positions, right=True) - 1
    within_idx = positions - qo_indptr[batch_idx]
    lse = lse_packed[batch_idx, :, within_idx] * inv_log2

    return output, lse
