import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    half_dim = emb.shape[-1] // 2
    grad_cos = grad_cos.float()
    grad_sin = grad_sin.float()
    grad_cos_freqs = grad_cos[..., :half_dim] + grad_cos[..., half_dim:]
    grad_sin_freqs = grad_sin[..., :half_dim] + grad_sin[..., half_dim:]
    emb_half = emb[..., :half_dim]
    grad_emb = (
        grad_cos_freqs * (-emb_half.sin()) * attention_scaling
        + grad_sin_freqs * emb_half.cos() * attention_scaling
    )
    return (inv_freq_expanded.transpose(-2, -1) @ grad_emb.transpose(1, 2)).squeeze(1)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    return _compiled(grad_cos, grad_sin, emb, inv_freq_expanded, attention_scaling)
