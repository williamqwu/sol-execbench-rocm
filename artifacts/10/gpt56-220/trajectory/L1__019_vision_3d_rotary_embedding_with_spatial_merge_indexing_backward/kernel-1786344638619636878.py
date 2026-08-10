import torch
import triton
import triton.language as tl


@triton.jit
def _qk_bwd(gq, gk, q, k, c, s, oq, ok, tc, ts,
            n: tl.constexpr, D: tl.constexpr = 128):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    pair = (d + 64) % 128
    base = row * D
    token = row // 16
    gvq = tl.load(gq + base + d)
    gvk = tl.load(gk + base + d)
    gpq = tl.load(gq + base + pair)
    gpk = tl.load(gk + base + pair)
    cv = tl.load(c + token * D + d)
    sp = tl.load(s + token * D + pair)
    sign = tl.where(d < 64, 1.0, -1.0)
    tl.store(oq + base + d, gvq * cv + sign * gpq * sp)
    tl.store(ok + base + d, gvk * cv + sign * gpk * sp)
    qv = tl.load(q + base + d)
    kv = tl.load(k + base + d)
    qp = tl.load(q + base + pair)
    kp = tl.load(k + base + pair)
    tl.store(tc + base + d, gvq * qv + gvk * kv)
    rsign = tl.where(d < 64, -1.0, 1.0)
    tl.store(ts + base + d, gvq * (rsign * qp) + gvk * (rsign * kp))


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, embeddings):
    # The reference concatenates embeddings with itself, evaluates trig over
    # both copies, and immediately discards the second copy.  Evaluate only
    # the live half while retaining its operation/reduction ordering.
    emb_cos = embeddings.cos()
    emb_sin = embeddings.sin()
    cos = emb_cos.unsqueeze(-2)
    sin = emb_sin.unsqueeze(-2)

    def rotate_half(x):
        x1 = x[..., :64]
        x2 = x[..., 64:]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_half_backward(g):
        g1 = g[..., :64]
        g2 = g[..., 64:]
        return torch.cat((g2, -g1), dim=-1)

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    tmp_cos = torch.empty_like(q)
    tmp_sin = torch.empty_like(q)
    _qk_bwd[(q.shape[0] * 16,)](grad_q_embed, grad_k_embed, q, k,
                                emb_cos, emb_sin, grad_q, grad_k,
                                tmp_cos, tmp_sin, q.shape[0], num_warps=2)

    grad_cos = tmp_cos.sum(dim=-2)
    grad_sin = tmp_sin.sum(dim=-2)
    grad_embeddings = torch.addcmul(grad_cos * (-emb_sin), grad_sin, emb_cos)
    return grad_q, grad_k, grad_embeddings
