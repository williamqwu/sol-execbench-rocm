import torch

_HALF = 64


@torch.compile(dynamic=True)
def _elementwise(grad_q_embed, grad_k_embed, cos, sin):
    cu = cos.unsqueeze(1)
    su = sin.unsqueeze(1)
    c1 = cu[..., :_HALF]; c2 = cu[..., _HALF:]
    s1 = su[..., :_HALF]; s2 = su[..., _HALF:]
    gqe1 = grad_q_embed[..., :_HALF]; gqe2 = grad_q_embed[..., _HALF:]
    gke1 = grad_k_embed[..., :_HALF]; gke2 = grad_k_embed[..., _HALF:]
    grad_q = torch.cat((gqe1 * c1 + gqe2 * s1, gqe2 * c2 - gqe1 * s2), dim=-1)
    grad_k = torch.cat((gke1 * c1 + gke2 * s1, gke2 * c2 - gke1 * s2), dim=-1)
    return grad_q, grad_k


@torch.compile(dynamic=True)
def _products(grad_q_embed, grad_k_embed, q, k):
    gqe1 = grad_q_embed[..., :_HALF]; gqe2 = grad_q_embed[..., _HALF:]
    q1 = q[..., :_HALF]; q2 = q[..., _HALF:]
    gke1 = grad_k_embed[..., :_HALF]; gke2 = grad_k_embed[..., _HALF:]
    k1 = k[..., :_HALF]; k2 = k[..., _HALF:]
    pq = grad_q_embed * q
    pk = grad_k_embed * k
    pqrot = torch.cat((gqe1 * (-q2), gqe2 * q1), dim=-1)
    pkrot = torch.cat((gke1 * (-k2), gke2 * k1), dim=-1)
    return pq, pk, pqrot, pkrot


@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    grad_q, grad_k = _elementwise(grad_q_embed, grad_k_embed, cos, sin)

    pq, pk, pqrot, pkrot = _products(grad_q_embed, grad_k_embed, q, k)
    grad_cos = pq.sum(1) + pk.sum(1)
    grad_sin = pqrot.sum(1) + pkrot.sum(1)

    return grad_q, grad_k, grad_cos, grad_sin
