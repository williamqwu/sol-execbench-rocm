import torch
import math


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    device = q.device
    batch_size = len_indptr - 1

    output = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (total_q, num_qo_heads), dtype=torch.float32, device=device
    )

    if batch_size == 0 or total_q == 0:
        output.fill_(0)
        lse.fill_(float("-inf"))
        return output, lse

    cum_seq_q = qo_indptr.to(torch.int32)
    cum_seq_k = kv_indptr.to(torch.int32)

    # For a single sequence (len_indptr == 2), max_seqlen == total_q exactly,
    # so no host sync is needed. For multi-sequence batches, compute the real
    # max (those workloads are compute-bound, so the sync is negligible).
    if len_indptr == 2:
        max_seqlen = total_q
    else:
        seqlens = cum_seq_q[1:] - cum_seq_q[:-1]
        max_seqlen = int(seqlens.max().item())

    out_f, lse_f, _, _, _ = torch.ops.aten._flash_attention_forward(
        q, k, v, cum_seq_q, cum_seq_k, max_seqlen, max_seqlen,
        0.0, True, False, scale=float(sm_scale),
    )
    output.copy_(out_f)

    inv_log2 = 1.0 / math.log(2.0)
    token_idx = torch.arange(total_q, device=device)
    batch_idx = torch.searchsorted(cum_seq_q[1:], token_idx, right=True)
    local_pos = token_idx - cum_seq_q[batch_idx]
    lse.copy_(lse_f[batch_idx, :, local_pos] * inv_log2)

    return output, lse
