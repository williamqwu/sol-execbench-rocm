import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch = q.shape[0]
    output = torch.zeros_like(q)
    lse = torch.full((batch, 32), -float("inf"), dtype=torch.float32, device=q.device)
    kf = k_cache[:, 0].float()
    vf = v_cache[:, 0].float()
    for b in range(batch):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if start == end:
            continue
        ids = kv_indices[start:end].long()
        k = kf[ids].permute(1, 0, 2)          # [4, T, 128]
        v = vf[ids].permute(1, 0, 2)
        qb = q[b].float().reshape(4, 8, 128)  # [4, 8, 128]
        logits = torch.matmul(qb, k.transpose(1, 2)) * sm_scale
        lse[b] = (torch.logsumexp(logits, dim=-1) / math.log(2.0)).reshape(32)
        output[b] = torch.matmul(torch.softmax(logits, dim=-1), v).reshape(32, 128).bfloat16()
    return output, lse
