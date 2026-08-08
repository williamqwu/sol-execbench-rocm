import torch
import math


def _attention_chunked(q_cat, K_cat, Kc_f, sm_scale, q_len, kv_len, prefix_len,
                       log2, chunk_kv, output, lse, q_start):
    """Online (flash) softmax over KV chunks for one batch.

    q_cat: [q_len, num_heads, 576] fp32
    K_cat: [kv_len, 576] fp32
    Kc_f:  [kv_len, head_dim_ckv] fp32
    """
    num_heads = q_cat.shape[1]
    device = q_cat.device

    m_i = torch.full((q_len, num_heads), -float("inf"), device=device, dtype=torch.float32)
    l_i = torch.zeros((q_len, num_heads), device=device, dtype=torch.float32)
    acc = torch.zeros((q_len, num_heads, Kc_f.shape[1]), device=device, dtype=torch.float32)

    row = torch.arange(q_len, device=device, dtype=torch.int64)
    q_abs = prefix_len + row  # [q_len] absolute position of each query

    for kv_off in range(0, kv_len, chunk_kv):
        kv_end = min(kv_off + chunk_kv, kv_len)
        kc = kv_end - kv_off
        Kc_chunk = K_cat[kv_off:kv_end]      # [kc, 576]
        V_chunk = Kc_f[kv_off:kv_end]        # [kc, head_dim_ckv]

        logits = (q_cat @ Kc_chunk.T) * sm_scale  # [q_len, num_heads, kc]

        # causal mask: kv position j is valid for query i if j <= q_abs[i]
        col = torch.arange(kv_off, kv_end, device=device, dtype=torch.int64)  # [kc]
        mask = col[None, :] > q_abs[:, None]  # [q_len, kc]
        logits.masked_fill_(mask[:, None, :], -float("inf"))

        m_j = logits.amax(dim=-1)  # [q_len, num_heads]
        m_new = torch.maximum(m_i, m_j)
        p = torch.exp(logits - m_new[:, :, None])  # [q_len, num_heads, kc]

        alpha = torch.exp(m_i - m_new)  # [q_len, num_heads]
        l_i = l_i * alpha + p.sum(dim=-1)
        acc = acc * alpha[:, :, None] + p @ V_chunk
        m_i = m_new

    # lse in base 2
    lse_q = (m_i + torch.log(l_i)) / log2  # handle l_i>0
    lse[q_start:q_start + q_len] = lse_q
    out = acc / l_i[:, :, None]
    output[q_start:q_start + q_len] = out.to(torch.bfloat16)


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

    Kc_all = ckv_cache.squeeze(1)  # bf16
    Kp_all = kpe_cache.squeeze(1)  # bf16

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
    # Chunk size in KV dimension (tokens). Tuned to keep [q,16,chunk] manageable.
    chunk_kv = 4096

    for b in range(batch_size):
        q_start = qo_indptr_cpu[b]
        q_end = qo_indptr_cpu[b + 1]
        page_beg = kv_indptr_cpu[b]
        page_end = kv_indptr_cpu[b + 1]

        if q_start >= q_end or page_beg >= page_end:
            continue

        q_len = q_end - q_start
        kv_len = page_end - page_beg

        tok_idx = kv_indices_long[page_beg:page_end]
        Kc = Kc_all[tok_idx]  # [kv_len, 512] bf16
        Kp = Kp_all[tok_idx]  # [kv_len, 64] bf16

        K_cat = torch.cat([Kc, Kp], dim=-1).to(torch.float32)  # [kv_len, 576]
        q_cat = torch.cat(
            [q_nope[q_start:q_end], q_pe[q_start:q_end]], dim=-1
        ).to(torch.float32)  # [q_len, 16, 576]
        Kc_f = Kc.to(torch.float32)  # [kv_len, 512]

        prefix_len = kv_len - q_len

        _attention_chunked(q_cat, K_cat, Kc_f, sm_scale, q_len, kv_len, prefix_len,
                           log2, chunk_kv, output, lse, q_start)

    return output, lse
