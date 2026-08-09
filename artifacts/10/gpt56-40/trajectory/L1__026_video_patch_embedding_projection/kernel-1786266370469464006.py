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
    """
    Video patch embedding projection.
    
    Args:
        video: (batch, frames, channels, height, width)
        patch_projection_weight: (hidden_size, in_channels, 1, patch_size, patch_size)
        patch_projection_bias: (hidden_size,)
        spatial_pos_embedding: (1, num_spatial_patches, hidden_size)
        temporal_pos_embedding: (1, num_frames, 1, hidden_size)
        norm_weight: (hidden_size,)
        norm_bias: (hidden_size,)
        eps: layer norm epsilon
    
    Returns:
        output: (batch, total_patches, hidden_size)
    """
    batch_size, frames, channels, height, width = video.shape
    hidden_size = patch_projection_weight.shape[0]
    patch_size = patch_projection_weight.shape[3]

    # The temporal kernel extent is one, so this is exactly a batch of 2-D
    # convolutions.  Keeping frames in the batch dimension gives MIOpen the
    # simpler convolution problem.
    video = video.reshape(batch_size * frames, channels, height, width)
    patches = F.conv2d(
        video, patch_projection_weight[:, :, 0], patch_projection_bias,
        stride=patch_size,
    )
    num_h, num_w = patches.shape[-2:]
    patches = patches.flatten(2).transpose(1, 2)
    patches = patches.reshape(batch_size, frames, num_h * num_w, hidden_size)
    
    # Add spatial positional embeddings (broadcast across frames)
    # spatial_pos_embedding: (1, num_spatial_patches, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + spatial_pos_embedding.unsqueeze(1)
    
    # Add temporal positional embeddings (broadcast across spatial patches)
    # temporal_pos_embedding: (1, num_frames, 1, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + temporal_pos_embedding
    
    patches = patches.reshape(batch_size, frames * num_h * num_w, hidden_size)
    return F.layer_norm(patches, (hidden_size,), norm_weight, norm_bias, eps)
