import torch

@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    half = 64
    cu = cos.unsqueeze(1)
    su = sin.unsqueeze(1)
    c1 = cu[..., :half]
    c2 = cu[..., half:]
    s1 = su[..., :half]
    s2 = su[..., half:]

    # grad_q = gqe*cos - rotate_half(gqe)*sin, expanded without copies:
    #   [:64] = gqe1*c1 + gqe2*s1
    #   [64:] = gqe2*c2 - gqe1*s2
    gqe1 = grad_q_embed[..., :half]
    gqe2 = grad_q_embed[..., half:]
    grad_q = torch.cat(
        (gqe1 * c1 + gqe2 * s1, gqe2 * c2 - gqe1 * s2), dim=-1
    )

    gke1 = grad_k_embed[..., :half]
    gke2 = grad_k_embed[..., half:]
    grad_k = torch.cat(
        (gke1 * c1 + gke2 * s1, gke2 * c2 - gke1 * s2), dim=-1
    )

    # grad_cos = (gqe*q).sum(1) + (gke*k).sum(1)
    grad_cos = (grad_q_embed * q).sum(1) + (grad_k_embed * k).sum(1)

    # grad_sin = (gqe*rotate_half(q)).sum(1) + (gke*rotate_half(k)).sum(1)
    # rotate_half(q) = cat(-q2, q1); product = cat(gqe1*(-q2), gqe2*q1)
    q1 = q[..., :half]
    q2 = q[..., half:]
    k1 = k[..., :half]
    k2 = k[..., half:]
    grad_sin = (
        torch.cat((grad_q_embed[..., :half] * (-q2), grad_q_embed[..., half:] * q1), dim=-1).sum(1)
        + torch.cat((grad_k_embed[..., :half] * (-k2), grad_k_embed[..., half:] * k1), dim=-1).sum(1)
    )

    return grad_q, grad_k, grad_cos, grad_sin
