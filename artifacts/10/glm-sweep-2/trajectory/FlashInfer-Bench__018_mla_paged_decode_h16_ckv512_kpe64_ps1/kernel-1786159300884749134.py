import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    device = q_nope.device

    # squeeze the page dim (page_size=1): [num_pages, head_dim]
    ckv = ckv_cache[:, 0, :]  # [num_pages, head_dim_ckv] bf16
    kpe = kpe_cache[:, 0, :]  # [num_pages, head_dim_kpe] bf16

    log2 = math.log(2.0)

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            output[b].zero_()
            continue

        L_tokens = page_end - page_beg
        if L_tokens <= 0:
            output[b].zero_()
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        Kc = ckv[tok_idx].to(torch.float32)  # [L_tokens, head_dim_ckv]
        Kp = kpe[tok_idx].to(torch.float32)  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / log2

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}
