import math
import torch

_INV_LN2 = 1.0 / math.log(2.0)


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q = q.shape[0]

    if qo_indptr.numel() == 2 and total_q != 0 and k.shape[0] != 0:
        qb = q.transpose(0, 1).unsqueeze(0)
        kb = k.transpose(0, 1).unsqueeze(0)
        vb = v.transpose(0, 1).unsqueeze(0)
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            qb, kb, vb, 0.0, True, False, scale=float(sm_scale)
        )
        output = result[0].squeeze(0).transpose(0, 1)
        result[1].mul_(_INV_LN2)
        lse = result[1].squeeze(0).transpose(0, 1)
        return output, lse

    qo, kv = torch.stack((qo_indptr, kv_indptr)).tolist()
    outputs = []
    lses = []
    for b in range(len(qo) - 1):
        qs, qe = qo[b], qo[b + 1]
        ks, ke = kv[b], kv[b + 1]
        if qs >= qe:
            continue
        if ks >= ke:
            outputs.append(torch.zeros_like(q[qs:qe]))
            lses.append(torch.full((qe - qs, 32), -float("inf"), device=q.device))
            continue
        qb = q[qs:qe].transpose(0, 1).unsqueeze(0)
        kb = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vb = v[ks:ke].transpose(0, 1).unsqueeze(0)
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            qb, kb, vb, 0.0, True, False, scale=float(sm_scale)
        )
        outputs.append(result[0].squeeze(0).transpose(0, 1))
        lses.append(result[1].squeeze(0).transpose(0, 1))
    return torch.cat(outputs), torch.cat(lses).mul_(_INV_LN2)
