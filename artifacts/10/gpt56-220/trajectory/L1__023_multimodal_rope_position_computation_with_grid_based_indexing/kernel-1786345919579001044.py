import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_images = axes_and_scalars["num_images"]
    total_tokens = axes_and_scalars["total_tokens"]
    num_position_embeddings = axes_and_scalars["num_position_embeddings"]
    hidden_size = axes_and_scalars["hidden_size"]
    head_dim = axes_and_scalars["head_dim"]
    spatial_merge_size = axes_and_scalars["spatial_merge_size"]
    rope_theta = 10000.0
    
    # Generate grid_thw values that sum to total_tokens
    # Each entry is [t, h, w] where h and w are divisible by spatial_merge_size
    # total_tokens = sum(t * h * w) for all entries
    grid_thw = torch.zeros((num_images, 3), dtype=torch.int64, device=device)
    
    remaining_tokens = total_tokens
    for i in range(num_images):
        if i == num_images - 1:
            # Last image gets remaining tokens
            # Find t, h, w such that t * h * w = remaining_tokens
            # Use simple factorization
            t = 1
            # Try to find h, w divisible by 2
            hw = remaining_tokens
            # Find factors
            h = 2
            while hw % h != 0 and h < hw:
                h += 2
            if hw % h == 0:
                w = hw // h
                if w % 2 != 0:
                    # Adjust
                    h = 4
                    while hw % h != 0 and h < hw:
                        h += 2
                    if hw % h == 0:
                        w = hw // h
            else:
                h = 2
                w = 2
                t = remaining_tokens // 4
                if t == 0:
                    t = 1
            grid_thw[i] = torch.tensor([t, h, w], dtype=torch.int64)
        else:
            # Generate random valid dimensions
            t = max(1, min(4, remaining_tokens // (num_images - i) // 16))
            h = max(2, min(34, (remaining_tokens // (num_images - i) // t) // 4)) // 2 * 2
            w = max(2, min(34, (remaining_tokens // (num_images - i) // t // h))) // 2 * 2
            if h < 2:
                h = 2
            if w < 2:
                w = 2
            tokens_this = t * h * w
            if tokens_this > remaining_tokens:
                t = 1
                h = 2
                w = 2
                tokens_this = 4
            grid_thw[i] = torch.tensor([t, h, w], dtype=torch.int64)
            remaining_tokens -= tokens_this
    
    # Position embedding weights
    pos_embed_weight = torch.randn(num_position_embeddings, hidden_size, dtype=torch.float32, device=device)
    
    # Inverse frequencies for rotary embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32, device=device) / head_dim))
    
    return {
        "grid_thw": grid_thw,
        "pos_embed_weight": pos_embed_weight,
        "inv_freq": inv_freq
    }

@torch.no_grad()
def run(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor, inv_freq: torch.Tensor):
    device = grid_thw.device
    dtype = pos_embed_weight.dtype
    
    # Constants
    spatial_merge_size = 2
    num_grid_per_side = 35
    hidden_size = pos_embed_weight.shape[1]
    head_dim = inv_freq.shape[0] * 4
    mrope_section = [24, 20, 20]
    
    # Grid metadata is tiny.  Copy it once instead of synchronizing separately
    # for every t/h/w scalar throughout all three loops below.
    grid_dims = grid_thw.tolist()
    
    # Part 1: Bilinear Interpolation Position Embeddings
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    interpolation_cache = {}
    
    for t, h, w in grid_dims:
        cached = interpolation_cache.get((h, w))
        if cached is not None:
            indices, weights = cached
        else:
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w, device=device)
        
            h_idxs_floor = h_idxs.long()
            w_idxs_floor = w_idxs.long()
            h_idxs_ceil = (h_idxs_floor + 1).clamp(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs_floor + 1).clamp(max=num_grid_per_side - 1)
        
            dh = h_idxs - h_idxs_floor.float()
            dw = w_idxs - w_idxs_floor.float()
        
            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side
        
            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]
        
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            interpolation_cache[(h, w)] = (indices, weights)
        
        for i in range(4):
            idx_list[i].append(indices[i])
            weight_list[i].append(weights[i])
    
    idx_tensor = torch.stack([torch.cat(parts) for parts in idx_list])
    weight_tensor = torch.stack([torch.cat(parts) for parts in weight_list]).to(dtype)
    
    # Accumulate the four bilinear samples directly, avoiding the large
    # [4, positions, hidden] gathered temporary.
    bag_indices = idx_tensor.T.contiguous()
    bag_weights = weight_tensor.T.contiguous()
    patch_pos_embeds = F.embedding_bag(
        bag_indices, pos_embed_weight, mode="sum",
        per_sample_weights=bag_weights,
    )
    
    patch_pos_embeds = patch_pos_embeds.split([h * w for _, h, w in grid_dims])
    
    patch_pos_embeds_permute = []
    merge_size = spatial_merge_size
    for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_dims):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size,
                          w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    
    # Part 2: Rotary Position Embeddings with MRoPE
    total_tokens = sum(t * h * w for t, h, w in grid_dims)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
    
    offset = 0
    for num_frames, height, width in grid_dims:
        merged_h, merged_w = height // merge_size, width // merge_size
        
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)
        
        num_tokens = coords.shape[0]
        pos_ids[offset:offset + num_tokens] = coords
        offset += num_tokens
    
    # Avoid synchronizing for max_hw and materializing an outer-product table:
    # gathering row r from that table is exactly r * inv_freq.
    embeddings = (pos_ids.to(torch.float32).unsqueeze(-1) * inv_freq).flatten(1)
    
    # All three source planes in the reference are expanded from `embeddings`.
    # The subsequent interleaving therefore only copies values onto themselves.
    emb = torch.cat((embeddings, embeddings), dim=-1)
    cos_embeddings = emb.cos()
    sin_embeddings = emb.sin()
    
    return patch_pos_embeds, cos_embeddings, sin_embeddings
