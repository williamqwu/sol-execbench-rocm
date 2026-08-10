import math
import torch

INV_LN2 = 1.0 / math.log(2.0)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    # Fetch each offsets array in one synchronization rather than synchronizing
    # separately for every endpoint of every ragged sequence.
    if qo_indptr.numel() == 2:
        # A one-sequence ragged tensor occupies the complete packed inputs.
        # Avoid launching/copying offsets merely to recover [0, total].
        qptr = (0, q.shape[0])
        kptr = (0, k.shape[0])
    else:
        qptr, kptr = torch.stack((qo_indptr, kv_indptr)).cpu().tolist()
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
        qq = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kk = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vv = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            qq, kk, vv, 0.0, True, scale=float(sm_scale)
        )
        output[qs:qe] = out.squeeze(0).transpose(0, 1)
        lse[qs:qe] = logsumexp.squeeze(0).transpose(0, 1) * INV_LN2
    return output, lse
