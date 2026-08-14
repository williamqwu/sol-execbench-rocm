import torch
import torch.nn.functional as F

_CL = torch.channels_last


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
    x = x.contiguous(memory_format=_CL)
    enc0 = F.conv2d(x, encoder_conv0_weight.contiguous(memory_format=_CL), encoder_conv0_bias, padding=1)
    down0 = F.conv2d(enc0, downsample_conv0_weight.contiguous(memory_format=_CL), downsample_conv0_bias, stride=2, padding=1)
    enc1 = F.conv2d(down0, encoder_conv1_weight.contiguous(memory_format=_CL), encoder_conv1_bias, padding=1)
    down1 = F.conv2d(enc1, downsample_conv1_weight.contiguous(memory_format=_CL), downsample_conv1_bias, stride=2, padding=1)
    enc2 = F.conv2d(down1, encoder_conv2_weight.contiguous(memory_format=_CL), encoder_conv2_bias, padding=1)
    feat = F.conv2d(enc2, bottleneck_weight.contiguous(memory_format=_CL), bottleneck_bias, padding=1)
    up0 = F.conv_transpose2d(feat, upsample_conv0_weight.contiguous(memory_format=_CL), upsample_conv0_bias, stride=2)
    skip0 = torch.cat([up0, enc1], dim=1)
    dec0 = F.conv2d(skip0, decoder_conv0_weight.contiguous(memory_format=_CL), decoder_conv0_bias, padding=1)
    up1 = F.conv_transpose2d(dec0, upsample_conv1_weight.contiguous(memory_format=_CL), upsample_conv1_bias, stride=2)
    skip1 = torch.cat([up1, enc0], dim=1)
    dec1 = F.conv2d(skip1, decoder_conv1_weight.contiguous(memory_format=_CL), decoder_conv1_bias, padding=1)
    output = F.conv2d(dec1, output_conv_weight.contiguous(memory_format=_CL), output_conv_bias, padding=1)
    return output.contiguous()
