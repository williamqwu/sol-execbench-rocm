import torch


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    half_dim = emb.shape[-1] // 2

    # Reuse the first halves of the converted gradient buffers.  The sequence
    # of floating-point operations remains the same as the reference.
    grad_cos_f = grad_cos.float()
    grad_sin_f = grad_sin.float()
    grad_cos_freqs = grad_cos_f[..., :half_dim]
    grad_sin_freqs = grad_sin_f[..., :half_dim]
    grad_cos_freqs.add_(grad_cos_f[..., half_dim:])
    grad_sin_freqs.add_(grad_sin_f[..., half_dim:])

    emb_half = emb[..., :half_dim]
    grad_cos_freqs.mul_(-emb_half.sin()).mul_(attention_scaling)
    grad_sin_freqs.mul_(emb_half.cos()).mul_(attention_scaling)
    grad_cos_freqs.add_(grad_sin_freqs)

    return torch.bmm(
        inv_freq_expanded.transpose(1, 2), grad_cos_freqs.transpose(1, 2)
    ).squeeze(1)
