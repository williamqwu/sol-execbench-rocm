import torch
import math


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1
    assert total_q == qo_indptr[-1].item()
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1)  # [num_pages, head_dim_ckv] bf16
    Kp_all = kpe_cache.squeeze(1)  # [num_pages, head_dim_kpe] bf16

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    qo_indptr_cpu = qo_indptr.cpu().tolist()
    kv_indptr_cpu = kv_indptr.cpu().tolist()
    kv_indices_long = kv_indices.to(torch.long)

    log2 = math.log(2.0)

    for b in range(batch_size):
        q_start = qo_indptr_cpu[b]
        q_end = qo_indptr_cpu[b + 1]
        page_beg = kv_indptr_cpu[b]
        page_end = kv_indptr_cpu[b + 1]

        if q_start >= q_end or page_beg >= page_end:
            continue

        q_len = q_end - q_start
        kv_len = page_end - page_beg

        tok_idx = kv_indices_long[page_beg:page_end]  # [kv_len]
        Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv] bf16
        Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe] bf16

        # Fuse: concat K dims -> single matmul in fp32
        q_cat = torch.cat(
            [q_nope[q_start:q_end], q_pe[q_start:q_end]], dim=-1
        ).to(torch.float32)  # [q_len, num_heads, 576]
        K_cat = torch.cat([Kc, Kp], dim=-1).to(torch.float32)  # [kv_len, 576]

        logits = (q_cat @ K_cat.T) * sm_scale  # [q_len, num_heads, kv_len]

        # Causal mask
        prefix_len = kv_len - q_len
        row = torch.arange(q_len, device=device, dtype=torch.int64)
        col = torch.arange(kv_len, device=device, dtype=torch.int64)
        mask = col[None, :] > (prefix_len + row[:, None])
        logits.masked_fill_(mask[:, None, :], -float("inf"))

        lse[q_start:q_end] = torch.logsumexp(logits, dim=-1) / log2
        attn = torch.softmax(logits, dim=-1)
        out = attn @ Kc.to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        output[q_start:q_end] = out.to(torch.bfloat16)

    return output, lse
