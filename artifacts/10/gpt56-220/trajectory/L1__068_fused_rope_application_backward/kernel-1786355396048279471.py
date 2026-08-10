import torch


@torch.compile(fullgraph=True, dynamic=False)
def _compiled(grad_q_embed, grad_k_embed, q, k, cos, sin):
    cu = cos.unsqueeze(1)
    su = sin.unsqueeze(1)

    def rotate(x):
        return torch.cat((-x[..., 64:], x[..., :64]), dim=-1)

    grad_q = grad_q_embed * cu - rotate(grad_q_embed) * su
    grad_k = grad_k_embed * cu - rotate(grad_k_embed) * su
    grad_cos = (grad_q_embed * q).sum(1) + (grad_k_embed * k).sum(1)
    grad_sin = (grad_q_embed * rotate(q)).sum(1) + (grad_k_embed * rotate(k)).sum(1)
    return grad_q, grad_k, grad_cos, grad_sin


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    return _compiled(grad_q_embed, grad_k_embed, q, k, cos, sin)
