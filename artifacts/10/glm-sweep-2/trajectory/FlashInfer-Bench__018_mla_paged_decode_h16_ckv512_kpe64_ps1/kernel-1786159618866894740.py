import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    device = q_nope.device
    log2 = math.log(2.0)

    ckv = ckv_cache[:, 0, :]  # [num_pages, head_dim_ckv] bf16
    kpe = kpe_cache[:, 0, :]  # [num_pages, head_dim_kpe] bf16

    # Move indptr to CPU to avoid per-iteration GPU syncs
    indptr_cpu = kv_indptr.cpu().tolist()

    # Gather ALL needed tokens in one shot
    tok_idx = kv_indices.to(torch.long)
    Kc_all = ckv[tok_idx].to(torch.float32)  # [num_kv_indices, head_dim_ckv]
    Kp_all = kpe[tok_idx].to(torch.float32)  # [num_kv_indices, head_dim_kpe]

    qn_all = q_nope.to(torch.float32)  # [batch_size, num_qo_heads, head_dim_ckv]
    qp_all = q_pe.to(torch.float32)  # [batch_size, num_qo_heads, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = indptr_cpu[b]
        page_end = indptr_cpu[b + 1]

        if page_beg >= page_end:
            output[b].zero_()
            continue

        Kc = Kc_all[page_beg:page_end]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[page_beg:page_end]  # [L_tokens, head_dim_kpe]
        qn = qn_all[b]  # [num_qo_heads, head_dim_ckv]
        qp = qp_all[b]  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / log2

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}
