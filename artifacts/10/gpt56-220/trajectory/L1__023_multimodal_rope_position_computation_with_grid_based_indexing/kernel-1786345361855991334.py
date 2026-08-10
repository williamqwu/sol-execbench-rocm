import torch
import reference


@torch.no_grad()
def run(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor, inv_freq: torch.Tensor):
    return reference.run(grid_thw, pos_embed_weight, inv_freq)
