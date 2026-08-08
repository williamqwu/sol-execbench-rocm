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

    # Fast path: single batch — return views directly, no allocation
    if len_indptr == 2:
        q_b = q.transpose(0, 1).unsqueeze(0)
        k_b = k.transpose(0, 1).unsqueeze(0)
        v_b = v.transpose(0, 1).unsqueeze(0)

        out_b, lse_b = torch._scaled_dot_product_flash_attention(
            q_b, k_b, v_b, scale=sm_scale, is_causal=True
        )[:2]

        output = out_b.squeeze(0).transpose(0, 1)
        lse = lse_b.squeeze(0).transpose(0, 1) * inv_log2
        return output, lse

    # Multi-batch: use varlen flash attention (single kernel launch)
    # Move indptr to CPU to compute sequence lengths
    qo_indptr_cpu = qo_indptr.cpu().tolist()
    kv_indptr_cpu = kv_indptr.cpu().tolist()

    # Compute sequence lengths and max
    batch_size = len_indptr - 1
    seq_lens = []
    for b in range(batch_size):
        qs, qe = qo_indptr_cpu[b], qo_indptr_cpu[b + 1]
        ks, ke = kv_indptr_cpu[b], kv_indptr_cpu[b + 1]
        nq = qe - qs
        nkv = ke - ks
        if nq > 0 and nkv > 0:
            seq_lens.append(nq)

    max_q = max(seq_lens) if seq_lens else 1

    # Varlen flash attention: processes all sequences in one kernel
    # q: [total_q, 32, 128], k: [total_kv, 8, 128], v: [total_kv, 8, 128]
    # cum_seq_q/k: [batch+1] int32
    # Returns: output [total_q, 32, 128], lse [batch, 32, max_q]
    result = torch.ops.aten._flash_attention_forward(
        q, k, v, qo_indptr, kv_indptr, max_q, max_q,
        0.0, True, False, scale=sm_scale
    )
    output = result[0]
    lse_packed = result[1]  # [batch, 32, max_q]

    # Unpack LSE: [batch, 32, max_q] -> [total_q, 32]
    # Build index vectors for vectorized gather
    batch_idx = torch.empty(total_q, dtype=torch.long, device=device)
    within_idx = torch.empty(total_q, dtype=torch.long, device=device)
    pos = 0
    for b in range(batch_size):
        qs, qe = qo_indptr_cpu[b], qo_indptr_cpu[b + 1]
        nq = qe - qs
        if nq > 0:
            batch_idx[pos:pos + nq] = b
            within_idx[pos:pos + nq] = torch.arange(nq, device=device)
            pos += nq

    lse = lse_packed[batch_idx, :, within_idx] * inv_log2

    return output, lse
