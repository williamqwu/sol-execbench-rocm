import math
import torch
import torch.nn.functional as F


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
        # Exercise the ROCm fused SDPA backend for the attention/value path.
        sdpa = F.scaled_dot_product_attention(
            q[b:b + 1, :, None, :],
            k[None], v[None],
            scale=float(sm_scale), enable_gqa=True,
        )
        output[b] = sdpa[0, :, 0].bfloat16()
    return output, lse
