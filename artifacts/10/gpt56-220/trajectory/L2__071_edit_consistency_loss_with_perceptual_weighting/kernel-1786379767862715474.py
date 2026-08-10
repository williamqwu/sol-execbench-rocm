import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def get_inputs(
    axes_and_scalars: dict, device: torch.device
) -> dict:
    """Generate inputs for the edit consistency loss."""
    batch_size = axes_and_scalars["batch_size"]
    height = axes_and_scalars["height"]
    width = axes_and_scalars["width"]
    channels = 3
    feat_dim_0 = 64
    feat_dim_1 = 128
    feat_dim_2 = 256
    num_scales = 3
    
    # Image tensors
    predicted = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    target = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    source = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    
    # Edit mask - binary values
    edit_mask = (torch.rand(batch_size, 1, height, width, device=device) > 0.5).float()
    
    # Extractor weights with proper initialization
    extractor0_conv1_weight = torch.randn(feat_dim_0, channels, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor0_conv1_bias = torch.zeros(feat_dim_0, dtype=torch.float32, device=device)
    extractor0_conv2_weight = torch.randn(feat_dim_0, feat_dim_0, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor0_conv2_bias = torch.zeros(feat_dim_0, dtype=torch.float32, device=device)
    
    extractor1_conv1_weight = torch.randn(feat_dim_1, feat_dim_0, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor1_conv1_bias = torch.zeros(feat_dim_1, dtype=torch.float32, device=device)
    extractor1_conv2_weight = torch.randn(feat_dim_1, feat_dim_1, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor1_conv2_bias = torch.zeros(feat_dim_1, dtype=torch.float32, device=device)
    
    extractor2_conv1_weight = torch.randn(feat_dim_2, feat_dim_1, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor2_conv1_bias = torch.zeros(feat_dim_2, dtype=torch.float32, device=device)
    extractor2_conv2_weight = torch.randn(feat_dim_2, feat_dim_2, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor2_conv2_bias = torch.zeros(feat_dim_2, dtype=torch.float32, device=device)
    
    # Scale weights
    scale_weights = torch.ones(num_scales, dtype=torch.float32, device=device) / num_scales
    
    # Scalar values
    pixel_loss_weight = 1.0
    perceptual_loss_weight = 0.5
    edit_region_weight_multiplier = 2.0
    
    return {
        "predicted": predicted,
        "target": target,
        "source": source,
        "edit_mask": edit_mask,
        "extractor0_conv1_weight": extractor0_conv1_weight,
        "extractor0_conv1_bias": extractor0_conv1_bias,
        "extractor0_conv2_weight": extractor0_conv2_weight,
        "extractor0_conv2_bias": extractor0_conv2_bias,
        "extractor1_conv1_weight": extractor1_conv1_weight,
        "extractor1_conv1_bias": extractor1_conv1_bias,
        "extractor1_conv2_weight": extractor1_conv2_weight,
        "extractor1_conv2_bias": extractor1_conv2_bias,
        "extractor2_conv1_weight": extractor2_conv1_weight,
        "extractor2_conv1_bias": extractor2_conv1_bias,
        "extractor2_conv2_weight": extractor2_conv2_weight,
        "extractor2_conv2_bias": extractor2_conv2_bias,
        "scale_weights": scale_weights,
        "pixel_loss_weight": pixel_loss_weight,
        "perceptual_loss_weight": perceptual_loss_weight,
        "edit_region_weight_multiplier": edit_region_weight_multiplier,
    }


@torch.no_grad()
@torch.compile(fullgraph=True)
def run(
    predicted: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    edit_mask: torch.Tensor,
    extractor0_conv1_weight: torch.Tensor,
    extractor0_conv1_bias: torch.Tensor,
    extractor0_conv2_weight: torch.Tensor,
    extractor0_conv2_bias: torch.Tensor,
    extractor1_conv1_weight: torch.Tensor,
    extractor1_conv1_bias: torch.Tensor,
    extractor1_conv2_weight: torch.Tensor,
    extractor1_conv2_bias: torch.Tensor,
    extractor2_conv1_weight: torch.Tensor,
    extractor2_conv1_bias: torch.Tensor,
    extractor2_conv2_weight: torch.Tensor,
    extractor2_conv2_bias: torch.Tensor,
    scale_weights: torch.Tensor,
    pixel_loss_weight: float,
    perceptual_loss_weight: float,
    edit_region_weight_multiplier: float,
):
    # Explicit Sobel stencil.  For this 1-input/2-output convolution, shifted
    # pointwise arithmetic is cheaper than dispatching a general convolution.
    padded_mask = F.pad(edit_mask, (1, 1, 1, 1))
    tl = padded_mask[:, :, :-2, :-2]
    tc = padded_mask[:, :, :-2, 1:-1]
    tr = padded_mask[:, :, :-2, 2:]
    ml = padded_mask[:, :, 1:-1, :-2]
    mr = padded_mask[:, :, 1:-1, 2:]
    bl = padded_mask[:, :, 2:, :-2]
    bc = padded_mask[:, :, 2:, 1:-1]
    br = padded_mask[:, :, 2:, 2:]
    edges_x = ((tr - tl) + 2.0 * (mr - ml)) + (br - bl)
    edges_y = ((bl - tl) + 2.0 * (bc - tc)) + (br - tr)
    edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
    
    # Compute spatial weights
    spatial_weights = torch.ones_like(edit_mask)
    spatial_weights = spatial_weights + (edit_region_weight_multiplier - 1.0) * edit_mask
    spatial_weights = spatial_weights + 0.5 * edge_magnitude
    
    # 1. Pixel-level reconstruction loss (L1)
    pixel_diff = torch.abs(predicted - target)
    weighted_pixel_loss = pixel_diff * spatial_weights
    pixel_loss = weighted_pixel_loss.mean()
    
    # 2. Extract perceptual features for predicted and target
    def extract_features(x, conv1_w, conv1_b, conv2_w, conv2_b):
        x = F.conv2d(x, conv1_w, conv1_b, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, conv2_w, conv2_b, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x
    
    # Run both images through each extractor in one batched invocation.  This
    # preserves the independent convolutions while avoiding duplicate launches.
    batch = predicted.shape[0]
    both = torch.cat((predicted, target), dim=0)
    both_feat0 = extract_features(both, extractor0_conv1_weight, extractor0_conv1_bias, extractor0_conv2_weight, extractor0_conv2_bias)
    pred_feat0, target_feat0 = both_feat0[:batch], both_feat0[batch:]
    weights0 = F.interpolate(spatial_weights, size=pred_feat0.shape[2:], mode='bilinear', align_corners=False)
    loss0 = (torch.abs(pred_feat0 - target_feat0) * weights0).mean() * torch.abs(scale_weights[0])

    both_feat1 = extract_features(both_feat0, extractor1_conv1_weight, extractor1_conv1_bias, extractor1_conv2_weight, extractor1_conv2_bias)
    pred_feat1, target_feat1 = both_feat1[:batch], both_feat1[batch:]
    weights1 = F.interpolate(spatial_weights, size=pred_feat1.shape[2:], mode='bilinear', align_corners=False)
    loss1 = (torch.abs(pred_feat1 - target_feat1) * weights1).mean() * torch.abs(scale_weights[1])

    both_feat2 = extract_features(both_feat1, extractor2_conv1_weight, extractor2_conv1_bias, extractor2_conv2_weight, extractor2_conv2_bias)
    pred_feat2, target_feat2 = both_feat2[:batch], both_feat2[batch:]
    weights2 = F.interpolate(spatial_weights, size=pred_feat2.shape[2:], mode='bilinear', align_corners=False)
    loss2 = (torch.abs(pred_feat2 - target_feat2) * weights2).mean() * torch.abs(scale_weights[2])

    # Keep the reference's left-to-right accumulation order.
    perceptual_loss = (torch.zeros((), dtype=torch.float32, device=predicted.device) + loss0) + loss1 + loss2
    
    # 3. Preservation loss for non-edited regions
    inverse_mask = 1.0 - edit_mask
    preservation_diff = torch.abs(predicted - source) * inverse_mask
    preservation_loss = preservation_diff.mean()
    
    # Combine all losses
    total_loss = (
        pixel_loss_weight * pixel_loss +
        perceptual_loss_weight * perceptual_loss +
        0.3 * preservation_loss
    )
    
    return total_loss
