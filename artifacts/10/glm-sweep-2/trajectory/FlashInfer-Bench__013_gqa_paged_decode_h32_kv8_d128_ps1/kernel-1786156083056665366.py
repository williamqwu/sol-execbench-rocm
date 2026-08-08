import torch
import math

_LOG2 = math.log(2.0)


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    gqa_ratio = num_qo_heads // num_kv_heads  # 4

    device = q.device

    output = torch.empty(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    k_flat = k_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim] bf16
    v_flat = v_cache.squeeze(1)  # [num_pages, num_kv_heads, head_dim] bf16

    kv_indptr_list = kv_indptr.cpu().tolist()

    for b in range(batch_size):
        ps = kv_indptr_list[b]
        pe = kv_indptr_list[b + 1]
        if ps >= pe:
            output[b].zero_()
            continue

        token_indices = kv_indices[ps:pe].to(torch.long)
        num_tokens = pe - ps

        k_batch = k_flat[token_indices].to(torch.float32)  # [num_tokens, 8, 128]
        v_batch = v_flat[token_indices].to(torch.float32)  # [num_tokens, 8, 128]
        q_batch = q[b].to(torch.float32)  # [32, 128]

        q_grouped = q_batch.view(num_kv_heads, gqa_ratio, head_dim)  # [8, 4, 128]
        k_grouped = k_batch.permute(1, 0, 2)  # [8, num_tokens, 128]
        v_grouped = v_batch.permute(1, 0, 2)  # [8, num_tokens, 128]

        # logits: [8, 4, num_tokens]
        logits = torch.bmm(q_grouped, k_grouped.transpose(-1, -2)) * sm_scale

        lse[b] = (torch.logsumexp(logits, dim=-1) / _LOG2).view(-1)

        attn = torch.softmax(logits, dim=-1)  # [8, 4, num_tokens]
        out = torch.bmm(attn, v_grouped)  # [8, 4, 128]
        output[b] = out.view(num_qo_heads, head_dim).to(torch.bfloat16)

    return output, lse
