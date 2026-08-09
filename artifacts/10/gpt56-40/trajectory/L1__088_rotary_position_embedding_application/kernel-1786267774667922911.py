import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x, cos, sin, out, n_elements: tl.constexpr, seq_len: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dim = offs & 127
    pos = (offs // 128) % seq_len
    pair = tl.where(dim < 64, offs + 64, offs - 64)
    sign = tl.where(dim < 64, -1.0, 1.0)

    value = tl.load(x + offs, mask=mask)
    paired = tl.load(x + pair, mask=mask)
    c = tl.load(cos + pos * 128 + dim, mask=mask)
    s = tl.load(sin + pos * 128 + dim, mask=mask)
    tl.store(out + offs, value * c + (paired * sign) * s, mask=mask)


@torch.no_grad()
def run(query, key, cos, sin):
    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    nq = query.numel()
    nk = key.numel()
    seq = query.shape[2]
    _rope_kernel[(triton.cdiv(nq, 256),)](query, cos, sin, q_out, nq, seq,
                                          BLOCK=256, num_warps=4)
    _rope_kernel[(triton.cdiv(nk, 256),)](key, cos, sin, k_out, nk, seq,
                                          BLOCK=256, num_warps=4)
    return q_out, k_out
