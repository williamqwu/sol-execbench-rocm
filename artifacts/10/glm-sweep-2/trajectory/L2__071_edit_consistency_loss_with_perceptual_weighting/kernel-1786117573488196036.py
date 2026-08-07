import torch
import torch.nn.functional as F


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


# Cache Sobel kernels per device to avoid repeated host->device upload.
_SOBEL_CACHE: dict = {}


def _sobel(device: torch.device):
    k = _SOBEL_CACHE.get(device)
    if k is None:
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        k = (sobel_x, sobel_y)
        _SOBEL_CACHE[device] = k
    return k


@torch.no_grad()
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
    sobel_x, sobel_y = _sobel(edit_mask.device)

    # Compute edge map from edit mask
    edges_x = F.conv2d(edit_mask, sobel_x, padding=1)
    edges_y = F.conv2d(edit_mask, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(edges_x * edges_x + edges_y * edges_y + 1e-8)

    # Compute spatial weights
    spatial_weights = 1.0 + (edit_region_weight_multiplier - 1.0) * edit_mask + 0.5 * edge_magnitude

    # 1. Pixel-level reconstruction loss (L1)
    pixel_loss = (torch.abs(predicted - target) * spatial_weights).mean()

    # 2. Extract perceptual features. predicted and target share identical
    # extractor weights, so stack them along the batch dimension and run each
    # extractor once instead of twice. Conv/relu/maxpool are batch-independent,
    # so this is numerically identical to running them separately.
    def extract_features(x, conv1_w, conv1_b, conv2_w, conv2_b):
        x = F.conv2d(x, conv1_w, conv1_b, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, conv2_w, conv2_b, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    b = predicted.shape[0]
    x = torch.cat([predicted, target], dim=0)

    feat0 = extract_features(x, extractor0_conv1_weight, extractor0_conv1_bias, extractor0_conv2_weight, extractor0_conv2_bias)
    feat1 = extract_features(feat0, extractor1_conv1_weight, extractor1_conv1_bias, extractor1_conv2_weight, extractor1_conv2_bias)
    feat2 = extract_features(feat1, extractor2_conv1_weight, extractor2_conv1_bias, extractor2_conv2_weight, extractor2_conv2_bias)

    feat_list = [feat0, feat1, feat2]

    # Compute perceptual loss across scales
    perceptual_loss = torch.zeros((), dtype=torch.float32, device=predicted.device)
    sw_abs = torch.abs(scale_weights)

    for scale_idx in range(3):
        ft = feat_list[scale_idx]
        pred_feat = ft[:b]
        target_feat = ft[b:]

        downsampled_weights = F.interpolate(
            spatial_weights,
            size=(pred_feat.shape[2], pred_feat.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        scale_loss = (torch.abs(pred_feat - target_feat) * downsampled_weights).mean()
        perceptual_loss = perceptual_loss + scale_loss * sw_abs[scale_idx]

    # 3. Preservation loss for non-edited regions
    preservation_loss = (torch.abs(predicted - source) * (1.0 - edit_mask)).mean()

    # Combine all losses
    total_loss = (
        pixel_loss_weight * pixel_loss +
        perceptual_loss_weight * perceptual_loss +
        0.3 * preservation_loss
    )

    return total_loss


# Apply torch.compile to fuse the many elementwise ops and reduce kernel-launch
# overhead. The compiled callable replaces `run` so the harness picks it up via
# `kernel.run`. Compiled graphs persist across calls, so only the warmup
# iterations pay recompile cost; the timed loop runs the optimized graph.
_run_compiled = torch.compile(run, mode="default", dynamic=False)
run = _run_compiled
