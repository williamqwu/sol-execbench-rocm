import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(query, key, cos, sin, q_out, k_out,
                 Q_BLOCKS: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                 seq_len: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    is_q = pid < Q_BLOCKS
    offs = tl.where(is_q, pid, pid - Q_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
    qmask = is_q & (offs < NQ)
    kmask = (~is_q) & (offs < NK)
    mask = qmask | kmask
    dim = offs & 127
    pos = (offs // 128) % seq_len
    pair = tl.where(dim < 64, offs + 64, offs - 64)
    sign = tl.where(dim < 64, -1.0, 1.0)

    value = tl.load(query + offs, mask=qmask, other=0.0) + tl.load(key + offs, mask=kmask, other=0.0)
    paired = tl.load(query + pair, mask=qmask, other=0.0) + tl.load(key + pair, mask=kmask, other=0.0)
    c = tl.load(cos + pos * 128 + dim, mask=mask)
    s = tl.load(sin + pos * 128 + dim, mask=mask)
    result = value * c + (paired * sign) * s
    tl.store(q_out + offs, result, mask=qmask)
    tl.store(k_out + offs, result, mask=kmask)


@torch.no_grad()
def run(query, key, cos, sin):
    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    nq = query.numel()
    nk = key.numel()
    seq = query.shape[2]
    qb = triton.cdiv(nq, 256)
    kb = triton.cdiv(nk, 256)
    _rope_kernel[(qb + kb,)](query, key, cos, sin, q_out, k_out,
                             qb, nq, nk, seq, BLOCK=256, num_warps=4)
    return q_out, k_out
