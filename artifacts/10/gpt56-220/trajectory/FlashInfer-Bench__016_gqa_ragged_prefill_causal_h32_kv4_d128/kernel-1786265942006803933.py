import math
import torch

INV_LN2 = 1.0 / math.log(2.0)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]
    if qo_indptr.numel() == 2:
        # A one-sequence ragged tensor occupies the complete packed inputs.
        # Return the fused result's view directly, avoiding offset transfer,
        # output allocation, and a device-to-device layout copy.
        out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            0.0, True, scale=float(sm_scale),
        )
        return (out.squeeze(0).transpose(0, 1),
                logsumexp.squeeze(0).transpose(0, 1) * INV_LN2)

    # Fetch both offsets arrays in one synchronization rather than
    # synchronizing separately for every endpoint of every sequence.
    qptr, kptr = torch.stack((qo_indptr, kv_indptr)).cpu().tolist()
    output_parts = []
    lse_parts = []
    for b in range(len(qptr) - 1):
        qs, qe = qptr[b], qptr[b + 1]
        ks, ke = kptr[b], kptr[b + 1]
        if qs == qe or ks == ke:
            if qs != qe:
                output_parts.append(torch.zeros_like(q[qs:qe]))
                lse_parts.append(torch.full(
                    (qe - qs, 32), -float("inf"), dtype=torch.float32,
                    device=q.device))
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
        output_parts.append(out.squeeze(0).transpose(0, 1))
        lse_parts.append(logsumexp.squeeze(0).transpose(0, 1))
    return (torch.cat(output_parts, dim=0),
            torch.cat(lse_parts, dim=0) * INV_LN2)
