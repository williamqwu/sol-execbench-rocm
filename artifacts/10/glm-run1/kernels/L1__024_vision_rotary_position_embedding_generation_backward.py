import torch

@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def _compiled(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    emb: torch.Tensor,
    head_dim_half: int,
    head_dim_quarter: int,
):
    total_patches = grad_cos.shape[0]
    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
    grad_rot = grad_emb[:, :head_dim_half] + grad_emb[:, head_dim_half:]
    grad_freqs_indexed = grad_rot.reshape(total_patches, 2, head_dim_quarter)
    p0 = pos_ids[:, 0:1].to(torch.float32)
    p1 = pos_ids[:, 1:2].to(torch.float32)
    weighted = p0 * grad_freqs_indexed[:, 0, :] + p1 * grad_freqs_indexed[:, 1, :]
    return weighted.sum(0)

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor
) -> torch.Tensor:
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]
    return _compiled(grad_cos, grad_sin, pos_ids, emb, head_dim // 2, head_dim_quarter)
