import torch

@torch.no_grad()
def _run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    total_patches = grad_cos.shape[0]
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]

    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()

    grad_freqs_indexed = (
        grad_emb[:, : head_dim // 2] + grad_emb[:, head_dim // 2 :]
    ).reshape(total_patches, 2, head_dim_quarter)

    pos_ids_f = pos_ids.to(torch.float32)
    grad_inv_freq = (grad_freqs_indexed * pos_ids_f.unsqueeze(-1)).sum((0, 1))

    return grad_inv_freq


run = torch.compile(_run, dynamic=True, mode="reduce-overhead")
