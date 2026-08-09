import math
import torch


@torch.compile(dynamic=True)
def _grouped_attention(q, k, v, scale):
    # [Q, 4, 8, 128] x [K, 4, 128], retaining grouped heads throughout.
    qf = q.float().reshape(q.shape[0], 4, 8, 128)
    kf = k.float()
    vf = v.float()
    logits = torch.einsum("qgnd,kgd->qgnk", qf, kf) * scale
    qpos = torch.arange(q.shape[0], device=q.device)
    kpos = torch.arange(k.shape[0], device=q.device)
    delta = k.shape[0] - q.shape[0]
    logits = logits.masked_fill(kpos[None, :] >= qpos[:, None] + 1 + delta,
                                -float("inf"))
    lse = torch.logsumexp(logits, -1) / math.log(2.0)
    weights = torch.softmax(logits, -1)
    out = torch.einsum("qgnk,kgd->qgnd", weights, vf)
    return out.reshape(q.shape).to(torch.bfloat16), lse.reshape(q.shape[0], 32)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    output = torch.empty_like(q)
    lse = torch.empty((q.shape[0], 32), dtype=torch.float32, device=q.device)
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
        output[qs:qe], lse[qs:qe] = _grouped_attention(
            q[qs:qe], k[ks:ke], v[ks:ke], float(sm_scale))
    return output, lse
