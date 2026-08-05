import torch
import math

@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    merge_size = 2
    hidden_size = 1536
    hidden_size_expanded = 6144

    # Step 1: Layer normalization (pre-shuffle, on hidden_size dimension)
    hidden_norm = torch.nn.functional.layer_norm(
        hidden, (hidden_size,), ln_weight, ln_bias, eps
    )

    # Step 2: Spatial shuffle to merge patches
    grid_thw_cpu = grid_thw.cpu()
    gth = grid_thw_cpu.tolist()
    num_grids = len(gth)

    shuffled_patches = []
    offset = 0
    for i in range(num_grids):
        t, h, w = gth[i]
        num_patches_this = t * h * w
        patches = hidden_norm[offset:offset + num_patches_this]
        h_merged = h // merge_size
        w_merged = w // merge_size
        patches = patches.view(t, h_merged, merge_size, w_merged, merge_size, hidden_size)
        patches = patches.permute(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(t * h_merged * w_merged, hidden_size_expanded)
        shuffled_patches.append(patches)
        offset += num_patches_this

    hidden_shuffled = torch.cat(shuffled_patches, dim=0)

    # Step 3: Two-layer MLP with GELU
    hidden_fc1 = torch.nn.functional.linear(hidden_shuffled, fc1_weight, fc1_bias)
    hidden_gelu = torch.nn.functional.gelu(hidden_fc1)
    output = torch.nn.functional.linear(hidden_gelu, fc2_weight, fc2_bias)

    return output
