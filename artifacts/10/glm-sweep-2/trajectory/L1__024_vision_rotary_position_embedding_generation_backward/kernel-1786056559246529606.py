import torch

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for vision rotary position embedding generation.

    The reference scatters gradients into a [max_grid_size, Q] buffer via
    index_add_, then computes grad_inv_freq = seq^T @ grad_freqs.  Since
    seq[i] == i, this weighted sum collapses to a single reduction over patches
    that needs neither the host sync (pos_ids.max().item()) nor the scatter:

        grad_inv_freq[j] = sum_p pos_ids[p,0]*g[p,0,j] + pos_ids[p,1]*g[p,1,j]
    """
    total_patches = grad_cos.shape[0]
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]

    # Step 1: grad through cos/sin
    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()

    # Step 2: grad through concat (split and sum) + unflatten (view)
    grad_freqs_indexed = (
        grad_emb[:, : head_dim // 2] + grad_emb[:, head_dim // 2 :]
    ).reshape(total_patches, 2, head_dim_quarter)

    # Step 3+4 fused: weighted reduction over (patch, h/w) axes.
    pos_ids_f = pos_ids.to(torch.float32)
    grad_inv_freq = (grad_freqs_indexed * pos_ids_f.unsqueeze(-1)).sum((0, 1))

    return grad_inv_freq
