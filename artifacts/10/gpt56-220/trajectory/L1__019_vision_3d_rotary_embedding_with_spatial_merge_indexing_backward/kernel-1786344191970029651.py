import torch


def _impl(grad_q_embed, grad_k_embed, q, k, embeddings):
    c = embeddings.cos().unsqueeze(1)
    s = embeddings.sin().unsqueeze(1)

    gq1, gq2 = grad_q_embed.chunk(2, dim=-1)
    gk1, gk2 = grad_k_embed.chunk(2, dim=-1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    c1, c2 = c.chunk(2, dim=-1)
    s1, s2 = s.chunk(2, dim=-1)

    grad_q = torch.cat((gq1 * c1 + gq2 * s2,
                        gq2 * c2 - gq1 * s1), dim=-1)
    grad_k = torch.cat((gk1 * c1 + gk2 * s2,
                        gk2 * c2 - gk1 * s1), dim=-1)

    qrot = torch.cat((-q2, q1), dim=-1)
    krot = torch.cat((-k2, k1), dim=-1)
    grad_cos = (grad_q_embed * q + grad_k_embed * k).sum(dim=1)
    grad_sin = (grad_q_embed * qrot + grad_k_embed * krot).sum(dim=1)
    grad_embeddings = grad_cos * (-s.squeeze(1)) + grad_sin * c.squeeze(1)
    return grad_q, grad_k, grad_embeddings


_compiled = torch.compile(_impl, fullgraph=True, dynamic=True)


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, embeddings):
    return _compiled(grad_q_embed, grad_k_embed, q, k, embeddings)
