import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    num_kv_indices = kv_indices.shape[0]

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
        logits_max = logits_scaled.amax(dim=-1, keepdim=True)
        attn = torch.exp(logits_scaled - logits_max)
        denom = attn.sum(dim=-1, keepdim=True)
        attn = attn / denom
        lse = torch.log2(denom.squeeze(-1)) + logits_max.squeeze(-1) * math.log2(math.e)
        output = torch.bmm(attn, Kc).to(torch.bfloat16)
        return {"output": output, "lse": lse}

    if num_kv_indices == 0:
        return {
            "output": torch.zeros(
                (1, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
            ),
            "lse": torch.full(
                (1, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            ),
        }

    tok_idx = kv_indices.to(torch.long)
    Kc = Kc_all[tok_idx].to(torch.float32)
    Kp = Kp_all[tok_idx].to(torch.float32)
    qn = q_nope[0].to(torch.float32)
    qp = q_pe[0].to(torch.float32)
    logits_scaled = (
        torch.cat((qn, qp), dim=-1) @ torch.cat((Kc, Kp), dim=-1).T
    ) * sm_scale
    logits_max = logits_scaled.amax(dim=-1, keepdim=True)
    attn = torch.exp(logits_scaled - logits_max)
    denom = attn.sum(dim=-1, keepdim=True)
    attn = attn / denom
    lse = (
        torch.log2(denom.squeeze(-1)) + logits_max.squeeze(-1) * math.log2(math.e)
    ).unsqueeze(0)
    output = (attn @ Kc).to(torch.bfloat16).unsqueeze(0)
    return {"output": output, "lse": lse}
