import torch
import triton
import triton.language as tl


@triton.jit
def _rms_qk_kernel(q, k, wq, wk, oq, ok, n_rows: tl.constexpr,
                   eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offs = row * 128 + cols

    qv = tl.load(q + offs)
    kv = tl.load(k + offs)
    q_inv = tl.rsqrt(tl.sum(qv * qv, axis=0) * (1.0 / 128.0) + eps)
    k_inv = tl.rsqrt(tl.sum(kv * kv, axis=0) * (1.0 / 128.0) + eps)

    head_offs = (row % 48) * 128 + cols
    q_weight = tl.load(wq + head_offs)
    k_weight = tl.load(wk + head_offs)
    tl.store(oq + offs, qv * q_inv * q_weight)
    tl.store(ok + offs, kv * k_inv * k_weight)


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor,
        weight_k: torch.Tensor, eps: float):
    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)
    rows = query.numel() // 128
    _rms_qk_kernel[(rows,)](
        query, key, weight_q, weight_k, out_q, out_k,
        rows, eps=eps, BLOCK=128, num_warps=4,
    )
    return out_q, out_k
