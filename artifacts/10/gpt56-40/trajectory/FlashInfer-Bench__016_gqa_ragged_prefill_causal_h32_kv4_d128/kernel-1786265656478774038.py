import math
import torch


@torch.compile(dynamic=True)
def _flash_segment(q, k, v, scale):
    out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        0.0, True, scale=scale,
    )
    return (out.squeeze(0).transpose(0, 1),
            logsumexp.squeeze(0).transpose(0, 1) * (1.0 / math.log(2.0)))


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    # Fetch each offsets array in one synchronization rather than synchronizing
    # separately for every endpoint of every ragged sequence.
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

        # ROCm's fused attention accepts GQA directly: Q has 32 heads while
        # K/V retain their four physical heads. It also returns natural-log
        # LSE, avoiding both quadratic intermediates and a second attention.
        output[qs:qe], lse[qs:qe] = _flash_segment(
            q[qs:qe], k[ks:ke], v[ks:ke], float(sm_scale))
    return output, lse
