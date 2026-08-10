import math
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.full(
        (batch_size, 32), -float("inf"), dtype=torch.float32, device=q.device
    )

    # Convert once, as in the reference, then perform all four query heads of
    # each KV head together.  Shapes inside the loop are [8, 4, ...].
    k_cache = k_cache[:, 0].float()
    v_cache = v_cache[:, 0].float()
    q = q.float().view(batch_size, 8, 4, 128)
    offsets = kv_indptr.tolist()

    for b in range(batch_size):
        start = offsets[b]
        end = offsets[b + 1]
        if start == end:
            output[b].zero_()
            continue
        indices = kv_indices[start:end].long()
        k = k_cache[indices].permute(1, 2, 0)  # [8, 128, tokens]
        v = v_cache[indices].permute(1, 0, 2)  # [8, tokens, 128]
        logits = torch.bmm(q[b], k) * sm_scale
        lse[b] = (torch.logsumexp(logits, dim=-1) / math.log(2.0)).reshape(32)
        output[b] = torch.bmm(torch.softmax(logits, dim=-1), v).reshape(32, 128)

    return output, lse
