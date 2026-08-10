import torch


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

    grad_q = grad_q_embed * cos + rotate_half_backward(grad_q_embed * sin)
    grad_k = grad_k_embed * cos + rotate_half_backward(grad_k_embed * sin)

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
