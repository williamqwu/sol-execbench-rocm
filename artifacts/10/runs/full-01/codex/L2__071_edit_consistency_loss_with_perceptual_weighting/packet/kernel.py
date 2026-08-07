import torch
import torch.nn.functional as F


@torch.library.custom_op("sol071::conv_relu", mutates_args=())
def _conv_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.miopen_convolution_relu(
        x, weight, bias, [1, 1], [1, 1], [1, 1], 1
    )


@_conv_relu.register_fake
def _conv_relu_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0], x.shape[2], x.shape[3]))


@torch.no_grad()
def run(
    predicted,
    target,
    source,
    edit_mask,
    extractor0_conv1_weight,
    extractor0_conv1_bias,
    extractor0_conv2_weight,
    extractor0_conv2_bias,
    extractor1_conv1_weight,
    extractor1_conv1_bias,
    extractor1_conv2_weight,
    extractor1_conv2_bias,
    extractor2_conv1_weight,
    extractor2_conv1_bias,
    extractor2_conv2_weight,
    extractor2_conv2_bias,
    scale_weights,
    pixel_loss_weight,
    perceptual_loss_weight,
    edit_region_weight_multiplier,
):
    sobel = torch.tensor(
        [
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        ],
        dtype=torch.float32,
        device=edit_mask.device,
    )

    edges = F.conv2d(edit_mask, sobel, padding=1)
    edges_x = edges[:, 0:1]
    edges_y = edges[:, 1:2]
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)

    spatial_weights = torch.ones_like(edit_mask)
    spatial_weights = spatial_weights + (edit_region_weight_multiplier - 1.0) * edit_mask
    spatial_weights = spatial_weights + 0.5 * edge_magnitude

    pixel_diff = torch.abs(predicted - target)
    weighted_pixel_loss = pixel_diff * spatial_weights
    pixel_loss = weighted_pixel_loss.mean()

    def conv_relu(x, weight, bias, fused):
        if fused:
            return _conv_relu(x, weight, bias)
        return F.relu(F.conv2d(x, weight, bias, padding=1))

    def extract_features(
        x, conv1_w, conv1_b, conv2_w, conv2_b, fused1=True, fused2=True
    ):
        x = conv_relu(x, conv1_w, conv1_b, fused1)
        x = conv_relu(x, conv2_w, conv2_b, fused2)
        return F.max_pool2d(x, kernel_size=2, stride=2)

    batch_size = predicted.shape[0]
    image_pair = torch.cat((predicted, target), dim=0)
    large_workload = batch_size * predicted.shape[2] * predicted.shape[3] >= 700_000
    if large_workload:
        fusion_flags = ((True, False), (True, False), (False, False))
    else:
        fusion_flags = ((True, True), (True, True), (True, True))
    feat0 = extract_features(
        image_pair,
        extractor0_conv1_weight,
        extractor0_conv1_bias,
        extractor0_conv2_weight,
        extractor0_conv2_bias,
        *fusion_flags[0],
    )
    feat1 = extract_features(
        feat0,
        extractor1_conv1_weight,
        extractor1_conv1_bias,
        extractor1_conv2_weight,
        extractor1_conv2_bias,
        *fusion_flags[1],
    )
    feat2 = extract_features(
        feat1,
        extractor2_conv1_weight,
        extractor2_conv1_bias,
        extractor2_conv2_weight,
        extractor2_conv2_bias,
        *fusion_flags[2],
    )

    perceptual_loss = torch.tensor(0.0, dtype=torch.float32, device=predicted.device)
    for scale_idx, feat in enumerate((feat0, feat1, feat2)):
        pred_feat = feat[:batch_size]
        target_feat = feat[batch_size:]
        factor = 2 ** (scale_idx + 1)
        offset = factor // 2 - 1
        w00 = spatial_weights[:, :, offset::factor, offset::factor]
        w01 = spatial_weights[:, :, offset::factor, offset + 1 :: factor]
        w10 = spatial_weights[:, :, offset + 1 :: factor, offset::factor]
        w11 = spatial_weights[:, :, offset + 1 :: factor, offset + 1 :: factor]
        downsampled_weights = ((w00 + w01) * 0.5 + (w10 + w11) * 0.5) * 0.5
        feat_diff = torch.abs(pred_feat - target_feat)
        weighted_feat_diff = feat_diff * downsampled_weights
        scale_loss = weighted_feat_diff.mean()
        scale_loss = scale_loss * torch.abs(scale_weights[scale_idx])
        perceptual_loss = perceptual_loss + scale_loss

    inverse_mask = 1.0 - edit_mask
    preservation_diff = torch.abs(predicted - source) * inverse_mask
    preservation_loss = preservation_diff.mean()

    return (
        pixel_loss_weight * pixel_loss
        + perceptual_loss_weight * perceptual_loss
        + 0.3 * preservation_loss
    )


run = torch.compile(
    run,
    dynamic=True,
    fullgraph=True,
    options={"max_autotune": False},
)
