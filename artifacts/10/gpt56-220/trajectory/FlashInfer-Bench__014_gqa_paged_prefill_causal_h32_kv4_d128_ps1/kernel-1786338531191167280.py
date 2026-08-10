import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    output = torch.zeros_like(q)
    lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32,
                     device=q.device)

    # One synchronization for all segment metadata, rather than two .item()
    # synchronizations for every sequence.
    qptr = qo_indptr.tolist()
    kptr = kv_indptr.tolist()
    kflat = k_cache[:, 0]
    vflat = v_cache[:, 0]
    scale = float(sm_scale)

    for b in range(len(qptr) - 1):
        qs, qe = qptr[b], qptr[b + 1]
        ks, ke = kptr[b], kptr[b + 1]
        nq, nk = qe - qs, ke - ks
        if nq <= 0 or nk <= 0:
            continue
        pages = kv_indices[ks:ke].long()
        kb = kflat[pages].float()
        vb = vflat[pages].float()
        qb = q[qs:qe].float().reshape(nq, 4, 8, 128)
        delta = nk - nq
        # [q,k] mask is shared by all heads.
        qi = torch.arange(nq, device=q.device)[:, None]
        ki = torch.arange(nk, device=q.device)[None, :]
        mask = ki <= qi + delta
        valid_q = (qi[:, 0] + delta) >= 0

        for kh in range(4):
            # Eight GQA heads are a batched GEMM, allowing rocBLAS/MFMA use.
            qh = qb[:, kh].permute(1, 0, 2)             # [8,q,128]
            logits = torch.matmul(qh, kb[:, kh].T) * scale  # [8,q,k]
            logits.masked_fill_(~mask[None], -float("inf"))
            lh = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            probs = torch.softmax(logits, dim=-1)
            # Avoid the NaNs produced by softmax of an entirely masked row.
            probs[:, ~valid_q] = 0
            oh = torch.matmul(probs, vb[:, kh])
            output[qs:qe, kh * 8:(kh + 1) * 8] = oh.permute(1, 0, 2).to(torch.bfloat16)
            lse[qs:qe, kh * 8:(kh + 1) * 8] = torch.where(
                valid_q[:, None], lh.T, -float("inf"))
    return output, lse
