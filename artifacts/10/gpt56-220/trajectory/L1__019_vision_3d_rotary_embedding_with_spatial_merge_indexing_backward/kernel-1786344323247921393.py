import torch
import triton
import triton.language as tl


@triton.jit
def _qk_bwd(gq, gk, c, s, oq, ok, n: tl.constexpr, D: tl.constexpr = 128):
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
    _qk_bwd[(q.shape[0] * 16,)](grad_q_embed, grad_k_embed, emb_cos, emb_sin,
                                grad_q, grad_k, q.shape[0], num_warps=2)

    grad_cos_q = grad_q_embed * q
    grad_cos_k = grad_k_embed * k
    grad_cos = (grad_cos_q + grad_cos_k).sum(dim=-2)
    gq1, gq2 = grad_q_embed[..., :64], grad_q_embed[..., 64:]
    gk1, gk2 = grad_k_embed[..., :64], grad_k_embed[..., 64:]
    q1, q2 = q[..., :64], q[..., 64:]
    k1, k2 = k[..., :64], k[..., 64:]
    grad_sin = torch.cat(((gq1 * (-q2) + gk1 * (-k2)).sum(dim=-2),
                          (gq2 * q1 + gk2 * k1).sum(dim=-2)), dim=-1)
    grad_embeddings = grad_cos * (-emb_sin) + grad_sin * emb_cos
    return grad_q, grad_k, grad_embeddings
