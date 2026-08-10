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

    grad_q = grad_q_embed * cos_unsqueezed
    grad_q_rotated = rotate_half(grad_q_embed)
    grad_q_rotated.mul_(sin_unsqueezed)
    grad_q.sub_(grad_q_rotated)

    grad_k = grad_k_embed * cos_unsqueezed
    grad_k_rotated = rotate_half(grad_k_embed)
    grad_k_rotated.mul_(sin_unsqueezed)
    grad_k.sub_(grad_k_rotated)

    grad_cos_from_q = grad_q_embed * q
    grad_cos_from_k = grad_k_embed * k
    grad_cos = grad_cos_from_q.sum(dim=unsqueeze_dim) + grad_cos_from_k.sum(dim=unsqueeze_dim)

    # Form the two halves directly.  This avoids materializing the full
    # rotated q/k tensors while retaining the same per-head reductions.
    grad_sin_1 = (grad_q_embed[..., :64] * q[..., 64:]).sum(dim=unsqueeze_dim)
    grad_sin_1 += (grad_k_embed[..., :64] * k[..., 64:]).sum(dim=unsqueeze_dim)
    grad_sin_1.neg_()
    grad_sin_2 = (grad_q_embed[..., 64:] * q[..., :64]).sum(dim=unsqueeze_dim)
    grad_sin_2 += (grad_k_embed[..., 64:] * k[..., :64]).sum(dim=unsqueeze_dim)
    grad_sin = torch.cat((grad_sin_1, grad_sin_2), dim=-1)

    return grad_q, grad_k, grad_cos, grad_sin
