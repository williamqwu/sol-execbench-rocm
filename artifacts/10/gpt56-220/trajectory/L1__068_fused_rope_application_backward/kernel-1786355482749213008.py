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

    grad_cos_from_q = grad_q_embed * q
    grad_cos_from_k = grad_k_embed * k
    grad_cos = grad_cos_from_q.sum(dim=unsqueeze_dim) + grad_cos_from_k.sum(dim=unsqueeze_dim)

    q_rotated = rotate_half(q)
    k_rotated = rotate_half(k)
    grad_sin_from_q = grad_q_embed * q_rotated
    grad_sin_from_k = grad_k_embed * k_rotated
    grad_sin = grad_sin_from_q.sum(dim=unsqueeze_dim) + grad_sin_from_k.sum(dim=unsqueeze_dim)

    return grad_q, grad_k, grad_cos, grad_sin
