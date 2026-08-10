import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode(
    Q, K, V, INDPTR, INDICES, OUT, LSE,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_kp: tl.constexpr, stride_kh: tl.constexpr,
    stride_vp: tl.constexpr, stride_vh: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    sm_scale: tl.constexpr, MAX_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kh = h // 8
    d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + b * stride_qb + h * stride_qh + d).to(tl.float32)
    begin = tl.load(INDPTR + b)
    end = tl.load(INDPTR + b + 1)
    m = -float("inf")
    denom = 0.0
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for base in range(0, MAX_TOKENS, BLOCK_N):
        n = base + tl.arange(0, BLOCK_N)
        valid = begin + n < end
        page = tl.load(INDICES + begin + n, mask=valid, other=0)
        k = tl.load(K + page[:, None] * stride_kp + kh * stride_kh + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * sm_scale
        score = tl.where(valid, score, -float("inf"))
        block_m = tl.max(score, axis=0)
        new_m = tl.maximum(m, block_m)
        old_scale = tl.exp(m - new_m)
        p = tl.exp(score - new_m)
        block_l = tl.sum(p, axis=0)
        v = tl.load(V + page[:, None] * stride_vp + kh * stride_vh + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * old_scale + tl.sum(p[:, None] * v, axis=0)
        denom = denom * old_scale + block_l
        m = new_m
    nonempty = begin < end
    tl.store(OUT + b * stride_ob + h * stride_oh + d, tl.where(nonempty, acc / denom, 0.0))
    lse = m * 1.4426950408889634 + tl.log2(denom)
    tl.store(LSE + b * 32 + h, tl.where(nonempty, lse, -float("inf")))


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((batch, 32), dtype=torch.float32, device=q.device)
    if batch == 1:
        max_tokens = triton.next_power_of_2(kv_indices.shape[0])
    else:
        longest = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())
        max_tokens = triton.next_power_of_2(longest)
    _paged_decode[(batch, 32)](
        q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(2), output.stride(0), output.stride(1),
        sm_scale=float(sm_scale), MAX_TOKENS=max_tokens, BLOCK_N=8, BLOCK_D=128,
        num_warps=4,
    )
    return output, lse
