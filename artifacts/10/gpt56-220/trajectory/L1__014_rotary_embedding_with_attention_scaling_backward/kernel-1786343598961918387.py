import torch


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    grad_cos = grad_cos.float()
    grad_sin = grad_sin.float()

    half_dim = emb.shape[-1] // 2
    grad_cos_freqs = grad_cos[..., :half_dim] + grad_cos[..., half_dim:]
    grad_sin_freqs = grad_sin[..., :half_dim] + grad_sin[..., half_dim:]

    emb_half = emb[..., :half_dim]
    grad_emb = (
        grad_cos_freqs * (-emb_half.sin()) * attention_scaling
        + grad_sin_freqs * emb_half.cos() * attention_scaling
    )

    grad_position_ids_expanded = (
        inv_freq_expanded.transpose(-2, -1) @ grad_emb.transpose(1, 2)
    )
    return grad_position_ids_expanded.squeeze(1)
