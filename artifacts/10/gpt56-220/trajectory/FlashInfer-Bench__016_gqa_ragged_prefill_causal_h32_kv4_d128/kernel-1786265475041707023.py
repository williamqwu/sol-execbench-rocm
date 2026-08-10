import math
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    # One transfer avoids two device synchronizations for every sequence.
    qptr = qo_indptr.cpu().tolist()
    kptr = kv_indptr.cpu().tolist()
    for b in range(len(qptr) - 1):
        qs, qe = qptr[b], qptr[b + 1]
        ks, ke = kptr[b], kptr[b + 1]
        if qs == qe or ks == ke:
            if qs != qe:
                output[qs:qe].zero_()
                lse[qs:qe].fill_(-float("inf"))
            continue
        qq = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kk = k[ks:ke].repeat_interleave(8, dim=1).transpose(0, 1).unsqueeze(0)
        vv = v[ks:ke].repeat_interleave(8, dim=1).transpose(0, 1).unsqueeze(0)
        out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            qq, kk, vv, 0.0, True, scale=float(sm_scale)
        )
        output[qs:qe] = out.squeeze(0).transpose(0, 1)
        lse[qs:qe] = logsumexp.squeeze(0).transpose(0, 1) / math.log(2.0)
    return output, lse
