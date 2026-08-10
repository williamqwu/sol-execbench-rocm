import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1)  # [num_pages, head_dim_kpe]

    if batch_size > 1:
        lengths = kv_indptr[1:] - kv_indptr[:-1]
        max_len = int(lengths.max().item())
        if max_len == 0:
            output = torch.zeros(
                (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )
            return {"output": output, "lse": lse}

        starts = kv_indptr[:-1]
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.int64), lengths
        )
        positions = torch.arange(num_kv_indices, device=device) - torch.repeat_interleave(
            starts.to(torch.int64), lengths
        )
        slots = batch_ids * max_len + positions
        page_grid = torch.zeros(batch_size * max_len, device=device, dtype=torch.int64)
        page_grid[slots] = kv_indices.to(torch.int64)
        page_grid = page_grid.view(batch_size, max_len)

        Kc = Kc_all[page_grid].to(torch.float32)
        Kp = Kp_all[page_grid].to(torch.float32)
        qn = q_nope.to(torch.float32)
        qp = q_pe.to(torch.float32)
        logits_scaled = (
            torch.bmm(qn, Kc.transpose(1, 2))
            + torch.bmm(qp, Kp.transpose(1, 2))
        ) * sm_scale
        valid = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        logits_scaled.masked_fill_(~valid[:, None, :], -float("inf"))
        lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits_scaled, dim=-1)
        output = torch.bmm(attn, Kc).to(torch.bfloat16)
        return {"output": output, "lse": lse}

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        pages = kv_indices[page_beg:page_end]
        # Derive kv_len from kv_indptr (for page_size=1, num_pages == num_tokens)
        L_tokens = page_end - page_beg

        if L_tokens <= 0 or pages.numel() == 0:
            output[b].zero_()
            continue

        # Pages are token indices for page_size=1
        tok_idx = pages[:L_tokens].to(torch.long)

        Kc = Kc_all[tok_idx].to(torch.float32)  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx].to(torch.float32)  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = torch.cat((qn, qp), dim=-1) @ torch.cat((Kc, Kp), dim=-1).T
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}
