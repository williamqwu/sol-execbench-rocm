import math
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]
    inv_ln2 = 1.0 / math.log(2.0)

    if qo_indptr.numel() == 2 and total_q != 0 and k.shape[0] != 0:
        qb = q.transpose(0, 1).unsqueeze(0)
        kb = k.transpose(0, 1).unsqueeze(0)
        vb = v.transpose(0, 1).unsqueeze(0)
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            qb, kb, vb, 0.0, True, False, scale=float(sm_scale)
        )
        output = result[0].squeeze(0).transpose(0, 1)
        lse = result[1].squeeze(0).transpose(0, 1) * inv_ln2
        return output, lse

    qo, kv = torch.stack((qo_indptr, kv_indptr)).tolist()
    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    for b in range(len(qo) - 1):
        qs, qe = qo[b], qo[b + 1]
        ks, ke = kv[b], kv[b + 1]
        if qs >= qe or ks >= ke:
            continue
        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            qb, kb, vb, 0.0, True, False, scale=float(sm_scale)
        )
        output[qs:qe] = result[0].squeeze(0).transpose(0, 1)
        lse[qs:qe] = result[1].squeeze(0).transpose(0, 1) * inv_ln2
    return output, lse
