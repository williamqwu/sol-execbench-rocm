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

    inv_log2 = 1.0 / math.log(2.0)

    # Move indptr to CPU once to avoid per-batch GPU sync
    qo_indptr_cpu = qo_indptr.cpu().tolist()
    kv_indptr_cpu = kv_indptr.cpu().tolist()

    # Pre-transpose to [1, H, total, D] layout so we can slice per batch
    # without transpose+copy each iteration
    q_t = q.transpose(0, 1).unsqueeze(0)  # [1, 32, total_q, 128]
    k_t = k.transpose(0, 1).unsqueeze(0)  # [1, 8, total_kv, 128]
    v_t = v.transpose(0, 1).unsqueeze(0)  # [1, 8, total_kv, 128]

    # Pre-transpose output views for direct write
    # output is [total_q, 32, 128] -> output_t [1, 32, total_q, 128]
    output_t = output.transpose(0, 1).unsqueeze(0)
    # lse is [total_q, 32] -> lse_t [1, 32, total_q]
    lse_t = lse.transpose(0, 1).unsqueeze(0)

    for b in range(len_indptr - 1):
        q_start = qo_indptr_cpu[b]
        q_end = qo_indptr_cpu[b + 1]

        kv_start = kv_indptr_cpu[b]
        kv_end = kv_indptr_cpu[b + 1]

        if q_start >= q_end or kv_start >= kv_end:
            continue

        q_b = q_t[:, :, q_start:q_end, :]
        k_b = k_t[:, :, kv_start:kv_end, :]
        v_b = v_t[:, :, kv_start:kv_end, :]

        out_b, lse_b = torch._scaled_dot_product_flash_attention(
            q_b, k_b, v_b, scale=sm_scale, is_causal=True
        )[:2]

        output_t[:, :, q_start:q_end, :] = out_b
        lse_t[:, :, q_start:q_end] = lse_b * inv_log2

    return output, lse
