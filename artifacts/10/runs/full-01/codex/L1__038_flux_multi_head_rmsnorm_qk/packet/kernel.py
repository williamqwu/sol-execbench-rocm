import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_qk_kernel(
    query,
    key,
    weight_q,
    weight_k,
    out_q,
    out_k,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 128)
    offset = row * 128 + col

    # Keeping Q and K in one wave lets the backend overlap their loads and
    # reductions, while a single wave exactly covers the 128-wide head.
    q = tl.load(query + offset)
    k = tl.load(key + offset)
    q_var = tl.sum(q * q, axis=0) / 128
    k_var = tl.sum(k * k, axis=0) / 128
    q_inv = tl.rsqrt(q_var + eps)
    k_inv = tl.rsqrt(k_var + eps)

    weight_offset = (row % 48) * 128 + col
    wq = tl.load(weight_q + weight_offset)
    wk = tl.load(weight_k + weight_offset)
    q_norm = q * q_inv
    k_norm = k * k_inv
    tl.store(out_q + offset, q_norm * wq, cache_modifier=".cs")
    tl.store(out_k + offset, k_norm * wk, cache_modifier=".cs")


@torch.no_grad()
def run(query, key, weight_q, weight_k, eps):
    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)
    rows = query.numel() // 128
    _rmsnorm_qk_kernel[(rows,)](
        query,
        key,
        weight_q,
        weight_k,
        out_q,
        out_k,
        eps=eps,
        num_warps=1,
    )
    return out_q, out_k
