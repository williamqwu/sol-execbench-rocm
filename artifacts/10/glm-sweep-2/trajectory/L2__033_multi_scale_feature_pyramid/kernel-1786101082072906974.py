import math

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    height = axes_and_scalars["height"]
    width = axes_and_scalars["width"]

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)

    return {
        "x": torch.randn(batch_size, 3, height, width, device=device, generator=g),
        "encoder_conv0_weight": kaiming_conv(32, 3, 3, 3),
        "encoder_conv0_bias": torch.randn(32, device=device, generator=g),
        "downsample_conv0_weight": kaiming_conv(64, 32, 3, 3),
        "downsample_conv0_bias": torch.randn(64, device=device, generator=g),
        "encoder_conv1_weight": kaiming_conv(64, 64, 3, 3),
        "encoder_conv1_bias": torch.randn(64, device=device, generator=g),
        "downsample_conv1_weight": kaiming_conv(128, 64, 3, 3),
        "downsample_conv1_bias": torch.randn(128, device=device, generator=g),
        "encoder_conv2_weight": kaiming_conv(128, 128, 3, 3),
        "encoder_conv2_bias": torch.randn(128, device=device, generator=g),
        "bottleneck_weight": kaiming_conv(128, 128, 3, 3),
        "bottleneck_bias": torch.randn(128, device=device, generator=g),
        "upsample_conv0_weight": kaiming_conv(128, 64, 2, 2),
        "upsample_conv0_bias": torch.randn(64, device=device, generator=g),
        "decoder_conv0_weight": kaiming_conv(64, 128, 3, 3),
        "decoder_conv0_bias": torch.randn(64, device=device, generator=g),
        "upsample_conv1_weight": kaiming_conv(64, 32, 2, 2),
        "upsample_conv1_bias": torch.randn(32, device=device, generator=g),
        "decoder_conv1_weight": kaiming_conv(32, 64, 3, 3),
        "decoder_conv1_bias": torch.randn(32, device=device, generator=g),
        "output_conv_weight": kaiming_conv(3, 32, 3, 3),
        "output_conv_bias": torch.randn(3, device=device, generator=g),
    }


@torch.compile(mode="reduce-overhead")
def _forward(
    x,
    encoder_conv0_weight, encoder_conv0_bias,
    downsample_conv0_weight, downsample_conv0_bias,
    encoder_conv1_weight, encoder_conv1_bias,
    downsample_conv1_weight, downsample_conv1_bias,
    encoder_conv2_weight, encoder_conv2_bias,
    bottleneck_weight, bottleneck_bias,
    upsample_conv0_weight, upsample_conv0_bias,
    decoder_conv0_weight, decoder_conv0_bias,
    upsample_conv1_weight, upsample_conv1_bias,
    decoder_conv1_weight, decoder_conv1_bias,
    output_conv_weight, output_conv_bias,
):
    enc0 = F.conv2d(x, encoder_conv0_weight, encoder_conv0_bias, padding=1)
    down0 = F.conv2d(enc0, downsample_conv0_weight, downsample_conv0_bias, stride=2, padding=1)
    enc1 = F.conv2d(down0, encoder_conv1_weight, encoder_conv1_bias, padding=1)
    down1 = F.conv2d(enc1, downsample_conv1_weight, downsample_conv1_bias, stride=2, padding=1)
    enc2 = F.conv2d(down1, encoder_conv2_weight, encoder_conv2_bias, padding=1)
    feat = F.conv2d(enc2, bottleneck_weight, bottleneck_bias, padding=1)
    up0 = F.conv_transpose2d(feat, upsample_conv0_weight, upsample_conv0_bias, stride=2)
    skip0 = torch.cat([up0, enc1], dim=1)
    dec0 = F.conv2d(skip0, decoder_conv0_weight, decoder_conv0_bias, padding=1)
    up1 = F.conv_transpose2d(dec0, upsample_conv1_weight, upsample_conv1_bias, stride=2)
    skip1 = torch.cat([up1, enc0], dim=1)
    dec1 = F.conv2d(skip1, decoder_conv1_weight, decoder_conv1_bias, padding=1)
    output = F.conv2d(dec1, output_conv_weight, output_conv_bias, padding=1)
    return output


@torch.no_grad()
def run(
    x: torch.Tensor,
    encoder_conv0_weight: torch.Tensor,
    encoder_conv0_bias: torch.Tensor,
    downsample_conv0_weight: torch.Tensor,
    downsample_conv0_bias: torch.Tensor,
    encoder_conv1_weight: torch.Tensor,
    encoder_conv1_bias: torch.Tensor,
    downsample_conv1_weight: torch.Tensor,
    downsample_conv1_bias: torch.Tensor,
    encoder_conv2_weight: torch.Tensor,
    encoder_conv2_bias: torch.Tensor,
    bottleneck_weight: torch.Tensor,
    bottleneck_bias: torch.Tensor,
    upsample_conv0_weight: torch.Tensor,
    upsample_conv0_bias: torch.Tensor,
    decoder_conv0_weight: torch.Tensor,
    decoder_conv0_bias: torch.Tensor,
    upsample_conv1_weight: torch.Tensor,
    upsample_conv1_bias: torch.Tensor,
    decoder_conv1_weight: torch.Tensor,
    decoder_conv1_bias: torch.Tensor,
    output_conv_weight: torch.Tensor,
    output_conv_bias: torch.Tensor,
):
    return _forward(
        x,
        encoder_conv0_weight, encoder_conv0_bias,
        downsample_conv0_weight, downsample_conv0_bias,
        encoder_conv1_weight, encoder_conv1_bias,
        downsample_conv1_weight, downsample_conv1_bias,
        encoder_conv2_weight, encoder_conv2_bias,
        bottleneck_weight, bottleneck_bias,
        upsample_conv0_weight, upsample_conv0_bias,
        decoder_conv0_weight, decoder_conv0_bias,
        upsample_conv1_weight, upsample_conv1_bias,
        decoder_conv1_weight, decoder_conv1_bias,
        output_conv_weight, output_conv_bias,
    )
