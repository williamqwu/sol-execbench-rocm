import torch
import math


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128

    device = q.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    log2 = math.log(2.0)

    # Move indptr to CPU once to avoid per-batch GPU sync
    qo_indptr_cpu = qo_indptr.cpu().tolist()
    kv_indptr_cpu = kv_indptr.cpu().tolist()

    for b in range(len_indptr - 1):
        q_start = qo_indptr_cpu[b]
        q_end = qo_indptr_cpu[b + 1]

        kv_start = kv_indptr_cpu[b]
        kv_end = kv_indptr_cpu[b + 1]

        if q_start >= q_end or kv_start >= kv_end:
            continue

        # q: [nq, 32, 128] -> [1, 32, nq, 128]
        q_b = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        # k: [nkv, 8, 128] -> [1, 8, nkv, 128]
        k_b = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        v_b = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)

        out_b, lse_b = torch._scaled_dot_product_flash_attention(
            q_b, k_b, v_b, scale=sm_scale, is_causal=True
        )[:2]

        # out_b: [1, 32, nq, 128] -> [nq, 32, 128]
        output[q_start:q_end] = out_b.squeeze(0).transpose(0, 1)
        # lse_b: [1, 32, nq] -> [nq, 32]
        lse[q_start:q_end] = (lse_b.squeeze(0).transpose(0, 1)) / log2

    return output, lse
