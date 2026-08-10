import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    device = q.device
    output = torch.zeros_like(q)
    lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

    kflat = k_cache[:, 0]
    vflat = v_cache[:, 0]
    inv_log2 = 1.0 / math.log(2.0)
    # One synchronization per compact metadata array, rather than four scalar
    # GPU synchronizations for every sequence in the batch.
    qo_offsets = qo_indptr.tolist()
    kv_offsets = kv_indptr.tolist()

    # A sequence has only eight independent KV heads.  Process all four query
    # heads sharing a KV head together, and all valid query positions together.
    for b in range(qo_indptr.numel() - 1):
        qs, qe = qo_offsets[b], qo_offsets[b + 1]
        ks, ke = kv_offsets[b], kv_offsets[b + 1]
        nq, nk = qe - qs, ke - ks
        if nq <= 0 or nk <= 0:
            continue

        pages = kv_indices[ks:ke].long()
        kb = kflat[pages].float()
        vb = vflat[pages].float()
        # With right-aligned causal attention, leading queries can have no KV.
        first = max(0, nq - nk)
        nv = nq - first
        if nv <= 0:
            continue
        row = torch.arange(first, nq, device=device)
        limit = row + 1 + nk - nq
        col = torch.arange(nk, device=device)
        causal = col[None, :] >= limit[:, None]

        # Materialize the GQA mapping once, then issue one strided batch GEMM
        # for all 32 heads instead of eight separate GEMMs.
        kh = kb.repeat_interleave(4, dim=1).permute(1, 2, 0)
        vh = vb.repeat_interleave(4, dim=1).permute(1, 0, 2)
        qh = q[qs + first:qe].float().permute(1, 0, 2)
        logits = torch.bmm(qh, kh)
        logits.mul_(sm_scale)
        logits.masked_fill_(causal[None, :, :], -float("inf"))
        lse[qs + first:qe] = (torch.logsumexp(logits, dim=-1) * inv_log2).T
        probs = torch.softmax(logits, dim=-1)
        out = torch.bmm(probs, vh).permute(1, 0, 2)
        output[qs + first:qe] = out.to(torch.bfloat16)

    return output, lse
