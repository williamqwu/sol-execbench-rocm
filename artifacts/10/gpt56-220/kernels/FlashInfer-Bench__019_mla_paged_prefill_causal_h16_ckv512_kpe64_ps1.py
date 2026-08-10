import torch
import math


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1)
    Kp_all = kpe_cache.squeeze(1)

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        kv_len = page_end - page_beg
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        tok_idx = pages[:kv_len].to(torch.long)
        # Gather the few live pages before widening.  Widening the full cache
        # touches roughly 1.1 GiB of unrelated data for every invocation.
        Kc = Kc_all[tok_idx].to(torch.float32)  # [kv_len, head_dim_ckv]
        Kp = Kp_all[tok_idx].to(torch.float32)  # [kv_len, head_dim_kpe]
        K = torch.cat((Kc, Kp), dim=-1)

        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        prefix_len = kv_len - q_len
        key_pos = torch.arange(kv_len, device=device)
        # A moderate chunk turns the token-at-a-time operations into GEMMs,
        # while bounding the [chunk, heads, kv_len] score allocation.
        for i in range(0, q_len, 64):
            j = min(i + 64, q_len)
            n = j - i
            qn = q_nope_batch[i:j].reshape(n * num_qo_heads, head_dim_ckv)
            qp = q_pe_batch[i:j].reshape(n * num_qo_heads, head_dim_kpe)
            q = torch.cat((qn, qp), dim=-1)
            logits = q @ K.T
            logits = logits.view(n, num_qo_heads, kv_len).mul_(sm_scale)
            query_pos = prefix_len + torch.arange(i, j, device=device)
            logits.masked_fill_(key_pos.view(1, 1, -1) > query_pos.view(-1, 1, 1), -float("inf"))

            lse[q_start + i:q_start + j] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1).view(n * num_qo_heads, kv_len)
            out = (attn @ Kc).view(n, num_qo_heads, head_dim_ckv)
            output[q_start + i:q_start + j] = out.to(torch.bfloat16)

    return output, lse
