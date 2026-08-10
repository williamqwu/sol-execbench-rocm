import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bwd(gq, gk, q, k, emb, oq, ok, oe, n_tokens: tl.constexpr,
              D: tl.constexpr = 128, H: tl.constexpr = 16,
              BH: tl.constexpr = 16, BD: tl.constexpr = 64):
    token = tl.program_id(0)
    half = tl.program_id(1)
    ds = half * BD + tl.arange(0, BD)
    hs = tl.arange(0, BH)
    offs = token * (H * D) + hs[:, None] * D + ds[None, :]
    pair_ds = (1 - half) * BD + tl.arange(0, BD)
    pair_offs = token * (H * D) + hs[:, None] * D + pair_ds[None, :]

    e = tl.load(emb + token * D + ds).to(tl.float32)
    ep = tl.load(emb + token * D + pair_ds).to(tl.float32)
    c = tl.cos(e)
    s = tl.sin(e)
    cp = tl.cos(ep)
    sp = tl.sin(ep)

    gqv = tl.load(gq + offs)
    gkv = tl.load(gk + offs)
    gqp = tl.load(gq + pair_offs)
    gkp = tl.load(gk + pair_offs)
    # R^T(g*sin): first half takes +second, second takes -first.
    sign = 1.0 - 2.0 * half
    tl.store(oq + offs, gqv * c[None, :] + sign * gqp * sp[None, :])
    tl.store(ok + offs, gkv * c[None, :] + sign * gkp * sp[None, :])

    qv = tl.load(q + offs)
    kv = tl.load(k + offs)
    qp = tl.load(q + pair_offs)
    kp = tl.load(k + pair_offs)
    # R(x): first half is -second, second half is +first.
    rsign = 2.0 * half - 1.0
    grad_cos = tl.sum(gqv * qv + gkv * kv, axis=0)
    grad_sin = tl.sum(gqv * (rsign * qp) + gkv * (rsign * kp), axis=0)
    tl.store(oe + token * D + ds, grad_cos * (-s) + grad_sin * c)


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, embeddings):
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    out_e = torch.empty_like(embeddings)
    n = q.shape[0]
    _rope_bwd[(n, 2)](grad_q_embed, grad_k_embed, q, k, embeddings,
                      out_q, out_k, out_e, n, num_warps=8)
    return out_q, out_k, out_e
