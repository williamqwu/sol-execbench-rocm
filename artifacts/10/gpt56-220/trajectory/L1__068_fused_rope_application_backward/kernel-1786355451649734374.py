import torch


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    cu = cos.unsqueeze(1)
    su = sin.unsqueeze(1)

    def grad_x(g):
        g1, g2 = g[..., :64], g[..., 64:]
        # Spell the first half exactly as subtract(negative(g2) * sin).
        return torch.cat((g1 * cu[..., :64] - (-g2) * su[..., :64],
                          g2 * cu[..., 64:] - g1 * su[..., 64:]), -1)

    grad_q = grad_x(grad_q_embed)
    grad_k = grad_x(grad_k_embed)
    grad_cos = (grad_q_embed * q).sum(1) + (grad_k_embed * k).sum(1)

    qr = torch.cat((-q[..., 64:], q[..., :64]), -1)
    kr = torch.cat((-k[..., 64:], k[..., :64]), -1)
    grad_sin = (grad_q_embed * qr).sum(1) + (grad_k_embed * kr).sum(1)
    return grad_q, grad_k, grad_cos, grad_sin
