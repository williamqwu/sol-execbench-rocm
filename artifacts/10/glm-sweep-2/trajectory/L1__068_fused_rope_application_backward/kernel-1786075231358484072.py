import torch

@torch.compile(mode="max-autotune", dynamic=True)
def _fwd(grad_q_embed, grad_k_embed, q, k, cos, sin):
    half_head_dim = 64
    unsqueeze_dim = 1
    cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
    sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half_head_dim]
        x2 = x[..., half_head_dim:]
        return torch.cat((-x2, x1), dim=-1)

    grad_q = (grad_q_embed * cos_unsqueezed) - (rotate_half(grad_q_embed) * sin_unsqueezed)
    grad_k = (grad_k_embed * cos_unsqueezed) - (rotate_half(grad_k_embed) * sin_unsqueezed)

    grad_cos = (grad_q_embed * q).sum(dim=unsqueeze_dim) + (grad_k_embed * k).sum(dim=unsqueeze_dim)

    grad_sin = (grad_q_embed * rotate_half(q)).sum(dim=unsqueeze_dim) + (grad_k_embed * rotate_half(k)).sum(dim=unsqueeze_dim)

    return grad_q, grad_k, grad_cos, grad_sin


@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    return _fwd(grad_q_embed, grad_k_embed, q, k, cos, sin)
