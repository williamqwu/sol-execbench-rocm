import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    device = q.device
    output = torch.zeros_like(q)
    lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

    qf = q.float()
    kflat = k_cache[:, 0]
    vflat = v_cache[:, 0]
    inv_log2 = 1.0 / math.log(2.0)

    # A sequence has only eight independent KV heads.  Process all four query
    # heads sharing a KV head together, and all valid query positions together.
    for b in range(qo_indptr.numel() - 1):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        ks = int(kv_indptr[b].item())
        ke = int(kv_indptr[b + 1].item())
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

        for kh in range(8):
            h0 = kh * 4
            qg = qf[qs + first:qe, h0:h0 + 4].reshape(nv * 4, 128)
            logits = torch.matmul(qg, kb[:, kh].T).reshape(nv, 4, nk)
            logits.mul_(sm_scale)
            logits.masked_fill_(causal[:, None, :], -float("inf"))
            lse[qs + first:qe, h0:h0 + 4] = torch.logsumexp(logits, dim=-1) * inv_log2
            probs = torch.softmax(logits, dim=-1).reshape(nv * 4, nk)
            out = torch.matmul(probs, vb[:, kh]).reshape(nv, 4, 128)
            output[qs + first:qe, h0:h0 + 4] = out.to(torch.bfloat16)

    return output, lse
