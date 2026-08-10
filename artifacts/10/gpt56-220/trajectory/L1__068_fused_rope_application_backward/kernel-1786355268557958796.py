import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bwd(gq, gk, q, k, cos, sin, oq, ok, oc, os,
              S: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // S
    s = row - b * S
    d = tl.arange(0, BLOCK)
    base = (b * S + s) * 128

    c1 = tl.load(cos + base + d)
    c2 = tl.load(cos + base + d + 64)
    s1 = tl.load(sin + base + d)
    s2 = tl.load(sin + base + d + 64)
    gc1 = tl.zeros((BLOCK,), tl.float32)
    gc2 = tl.zeros((BLOCK,), tl.float32)
    gs1 = tl.zeros((BLOCK,), tl.float32)
    gs2 = tl.zeros((BLOCK,), tl.float32)

    for h in tl.static_range(HQ):
        off = ((b * HQ + h) * S + s) * 128 + d
        a = tl.load(gq + off)
        z = tl.load(gq + off + 64)
        x = tl.load(q + off)
        y = tl.load(q + off + 64)
        tl.store(oq + off, a * c1 + z * s1)
        tl.store(oq + off + 64, z * c2 - a * s2)
        gc1 += a * x
        gc2 += z * y
        gs1 -= a * y
        gs2 += z * x

    for h in tl.static_range(HK):
        off = ((b * HK + h) * S + s) * 128 + d
        a = tl.load(gk + off)
        z = tl.load(gk + off + 64)
        x = tl.load(k + off)
        y = tl.load(k + off + 64)
        tl.store(ok + off, a * c1 + z * s1)
        tl.store(ok + off + 64, z * c2 - a * s2)
        gc1 += a * x
        gc2 += z * y
        gs1 -= a * y
        gs2 += z * x

    tl.store(oc + base + d, gc1)
    tl.store(oc + base + d + 64, gc2)
    tl.store(os + base + d, gs1)
    tl.store(os + base + d + 64, gs2)


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    b, hq, seq, _ = grad_q_embed.shape
    hk = grad_k_embed.shape[1]
    grad_q = torch.empty_like(grad_q_embed)
    grad_k = torch.empty_like(grad_k_embed)
    grad_cos = torch.empty_like(cos)
    grad_sin = torch.empty_like(sin)
    _rope_bwd[(b * seq,)](
        grad_q_embed, grad_k_embed, q, k, cos, sin,
        grad_q, grad_k, grad_cos, grad_sin,
        S=seq, HQ=hq, HK=hk, BLOCK=64, num_warps=1,
    )
    return grad_q, grad_k, grad_cos, grad_sin
