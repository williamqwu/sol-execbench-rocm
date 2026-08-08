import torch
import math

_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_GQA_RATIO = _NUM_QO_HEADS // _NUM_KV_HEADS  # 8
_LOG2 = math.log(2.0)


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 1
    assert len_indptr == batch_size + 1

    device = q.device
    inv_log2 = 1.0 / _LOG2

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    # Gather first (bf16), then convert only the gathered slice to fp32.
    # k_cache: [num_pages, 1, num_kv_heads, head_dim]
    k_squeezed = k_cache[:, 0]  # [num_pages, num_kv_heads, head_dim] bf16
    v_squeezed = v_cache[:, 0]  # [num_pages, num_kv_heads, head_dim] bf16

    kv_indices_long = kv_indices.to(torch.long)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_start >= page_end:
            output[b].zero_()
            continue

        token_indices = kv_indices_long[page_start:page_end]
        num_tokens = token_indices.shape[0]
        if num_tokens == 0:
            output[b].zero_()
            continue

        # Gather bf16 then convert: only num_tokens*4*128 elements
        k_batch = k_squeezed[token_indices].to(torch.float32)  # [num_tokens, num_kv_heads, head_dim]
        v_batch = v_squeezed[token_indices].to(torch.float32)
        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        k_heads = k_batch.permute(1, 0, 2)  # [num_kv_heads, num_tokens, head_dim]
        v_heads = v_batch.permute(1, 0, 2)

        k_exp = k_heads.repeat_interleave(_GQA_RATIO, dim=0)  # [num_qo_heads, num_tokens, head_dim]
        v_exp = v_heads.repeat_interleave(_GQA_RATIO, dim=0)

        q5 = q_batch.unsqueeze(1)  # [num_qo_heads, 1, head_dim]
        logits = torch.bmm(q5, k_exp.transpose(1, 2))  # [num_qo_heads, 1, num_tokens]
        logits = (logits * sm_scale).squeeze(1)  # [num_qo_heads, num_tokens]

        lse[b] = torch.logsumexp(logits, dim=-1) * inv_log2

        attn = torch.softmax(logits, dim=-1)  # [num_qo_heads, num_tokens]
        out = torch.bmm(attn.unsqueeze(1), v_exp).squeeze(1)  # [num_qo_heads, head_dim]
        output[b] = out.to(torch.bfloat16)

    return output, lse
