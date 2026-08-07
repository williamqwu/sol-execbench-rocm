import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    video: torch.Tensor,
    patch_projection_weight: torch.Tensor,
    patch_projection_bias: torch.Tensor,
    spatial_pos_embedding: torch.Tensor,
    temporal_pos_embedding: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
):
    batch_size, frames, channels, height, width = video.shape
    hidden_size = patch_projection_weight.shape[0]
    patch_size = patch_projection_weight.shape[3]
    
    # Rearrange to (batch, channels, frames, height, width) for Conv3D
    video = video.permute(0, 2, 1, 3, 4)
    
    # Apply patch projection using Conv3D
    patches = F.conv3d(
        video,
        patch_projection_weight,
        patch_projection_bias,
        stride=(1, patch_size, patch_size),
        padding=(0, 0, 0)
    )
    
    # Rearrange to (batch, frames, hidden_size, num_patches_h, num_patches_w)
    patches = patches.permute(0, 2, 1, 3, 4)
    
    batch_size, frames, hidden_size, num_h, num_w = patches.shape
    
    patches = patches.reshape(batch_size, frames, hidden_size, num_h * num_w)
    patches = patches.permute(0, 1, 3, 2)  # (B, F, num_spatial_patches, hidden_size)
    
    # Add positional embeddings
    patches = patches + spatial_pos_embedding.unsqueeze(1)
    patches = patches + temporal_pos_embedding
    
    # Flatten temporal and spatial dimensions: (B, F, S, H) -> (B, F*S, H)
    patches = patches.reshape(batch_size, frames * num_h * num_w, hidden_size)
    
    # Fused layer normalization
    output = F.layer_norm(patches, (hidden_size,), norm_weight, norm_bias, eps)
    
    return output
