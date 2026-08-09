import torch
import torch.nn.functional as F

def _fused_tail(patches, projection_bias, spatial, temporal, norm_weight, norm_bias, eps):
    patches = patches + projection_bias
    patches = patches + spatial.unsqueeze(1)
    patches = patches + temporal
    hidden_size = patches.shape[-1]
    patches = patches.reshape(patches.shape[0], -1, hidden_size)
    var, mean = torch.var_mean(patches, dim=-1, keepdim=True, unbiased=False)
    patches = (patches - mean) * torch.rsqrt(var + eps)
    return patches * norm_weight + norm_bias

@torch.no_grad()
@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={
        "shape_padding": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_check_all_directions": True,
        "coordinate_descent_search_radius": 2,
    },
)
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

    video = video.permute(0, 2, 1, 3, 4)
    patches = F.conv3d(
        video, patch_projection_weight, None,
        stride=(1, patch_size, patch_size), padding=(0, 0, 0),
    )
    patches = patches.permute(0, 2, 1, 3, 4)
    _, _, _, num_h, num_w = patches.shape
    patches = patches.reshape(batch_size, frames, hidden_size, num_h * num_w)
    patches = patches.permute(0, 1, 3, 2)
    
    return _fused_tail(
        patches, patch_projection_bias, spatial_pos_embedding, temporal_pos_embedding,
        norm_weight, norm_bias, eps,
    )
