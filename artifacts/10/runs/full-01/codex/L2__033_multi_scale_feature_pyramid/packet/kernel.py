import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _cat_pair_kernel(a, b, out, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    mask = offsets < CHUNK
    input_offsets = batch * CHUNK + offsets
    output_offsets = batch * (2 * CHUNK) + offsets
    values_a = tl.load(a + input_offsets, mask=mask)
    values_b = tl.load(b + input_offsets, mask=mask)
    tl.store(out + output_offsets, values_a, mask=mask)
    tl.store(out + output_offsets + CHUNK, values_b, mask=mask)


def _cat_equal(a, b):
    if a.numel() <= 4_000_000:
        return torch.cat((a, b), dim=1)
    out = torch.empty(
        (a.shape[0], 2 * a.shape[1], a.shape[2], a.shape[3]),
        device=a.device,
        dtype=a.dtype,
    )
    chunk = a.shape[1] * a.shape[2] * a.shape[3]
    _cat_pair_kernel[(triton.cdiv(chunk, 1024), a.shape[0])](
        a, b, out, CHUNK=chunk, BLOCK=1024, num_warps=8
    )
    return out


def _upsample_64_32(x, weight, bias):
    # With kernel == stride == 2, every input pixel writes an independent 2x2
    # output patch.  A 1x1 convolution computes its four phases in channels.
    phase_weight = (
        weight.permute(1, 2, 3, 0).reshape(32 * 4, 64, 1, 1).contiguous()
    )
    phase_bias = bias[:, None].expand(32, 4).reshape(32 * 4).contiguous()
    return F.pixel_shuffle(F.conv2d(x, phase_weight, phase_bias), 2)


def _network(args):
    (
        x,
        encoder_conv0_weight,
        encoder_conv0_bias,
        downsample_conv0_weight,
        downsample_conv0_bias,
        encoder_conv1_weight,
        encoder_conv1_bias,
        downsample_conv1_weight,
        downsample_conv1_bias,
        encoder_conv2_weight,
        encoder_conv2_bias,
        bottleneck_weight,
        bottleneck_bias,
        upsample_conv0_weight,
        upsample_conv0_bias,
        decoder_conv0_weight,
        decoder_conv0_bias,
        upsample_conv1_weight,
        upsample_conv1_bias,
        decoder_conv1_weight,
        decoder_conv1_bias,
        output_conv_weight,
        output_conv_bias,
    ) = args
    enc0 = F.conv2d(x, encoder_conv0_weight, encoder_conv0_bias, padding=1)
    down0 = F.conv2d(
        enc0, downsample_conv0_weight, downsample_conv0_bias, stride=2, padding=1
    )
    enc1 = F.conv2d(down0, encoder_conv1_weight, encoder_conv1_bias, padding=1)
    down1 = F.conv2d(
        enc1, downsample_conv1_weight, downsample_conv1_bias, stride=2, padding=1
    )
    enc2 = F.conv2d(down1, encoder_conv2_weight, encoder_conv2_bias, padding=1)
    feat = F.conv2d(enc2, bottleneck_weight, bottleneck_bias, padding=1)
    up0 = F.conv_transpose2d(
        feat, upsample_conv0_weight, upsample_conv0_bias, stride=2
    )
    dec0 = F.conv2d(
        _cat_equal(up0, enc1),
        decoder_conv0_weight,
        decoder_conv0_bias,
        padding=1,
    )
    if dec0.shape[2] >= 128:
        up1 = _upsample_64_32(dec0, upsample_conv1_weight, upsample_conv1_bias)
    else:
        up1 = F.conv_transpose2d(
            dec0, upsample_conv1_weight, upsample_conv1_bias, stride=2
        )
    dec1 = F.conv2d(
        _cat_equal(up1, enc0),
        decoder_conv1_weight,
        decoder_conv1_bias,
        padding=1,
    )
    return F.conv2d(dec1, output_conv_weight, output_conv_bias, padding=1)


@torch.no_grad()
def run(
    x,
    encoder_conv0_weight,
    encoder_conv0_bias,
    downsample_conv0_weight,
    downsample_conv0_bias,
    encoder_conv1_weight,
    encoder_conv1_bias,
    downsample_conv1_weight,
    downsample_conv1_bias,
    encoder_conv2_weight,
    encoder_conv2_bias,
    bottleneck_weight,
    bottleneck_bias,
    upsample_conv0_weight,
    upsample_conv0_bias,
    decoder_conv0_weight,
    decoder_conv0_bias,
    upsample_conv1_weight,
    upsample_conv1_bias,
    decoder_conv1_weight,
    decoder_conv1_bias,
    output_conv_weight,
    output_conv_bias,
):
    args = (
        x,
        encoder_conv0_weight,
        encoder_conv0_bias,
        downsample_conv0_weight,
        downsample_conv0_bias,
        encoder_conv1_weight,
        encoder_conv1_bias,
        downsample_conv1_weight,
        downsample_conv1_bias,
        encoder_conv2_weight,
        encoder_conv2_bias,
        bottleneck_weight,
        bottleneck_bias,
        upsample_conv0_weight,
        upsample_conv0_bias,
        decoder_conv0_weight,
        decoder_conv0_bias,
        upsample_conv1_weight,
        upsample_conv1_bias,
        decoder_conv1_weight,
        decoder_conv1_bias,
        output_conv_weight,
        output_conv_bias,
    )

    return _network(args)
