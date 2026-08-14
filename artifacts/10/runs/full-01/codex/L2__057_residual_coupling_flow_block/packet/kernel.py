import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    x,
    x_mask,
    reverse,
    transform_0_conv0_weight,
    transform_0_conv0_bias,
    transform_0_conv1_weight,
    transform_0_conv1_bias,
    transform_0_conv2_weight,
    transform_0_conv2_bias,
    transform_1_conv0_weight,
    transform_1_conv0_bias,
    transform_1_conv1_weight,
    transform_1_conv1_bias,
    transform_1_conv2_weight,
    transform_1_conv2_bias,
    transform_2_conv0_weight,
    transform_2_conv0_bias,
    transform_2_conv1_weight,
    transform_2_conv1_bias,
    transform_2_conv2_weight,
    transform_2_conv2_bias,
    transform_3_conv0_weight,
    transform_3_conv0_bias,
    transform_3_conv1_weight,
    transform_3_conv1_bias,
    transform_3_conv2_weight,
    transform_3_conv2_bias,
):
    transforms = (
        (transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias,
         transform_1_conv1_weight, transform_1_conv1_bias,
         transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias,
         transform_2_conv1_weight, transform_2_conv1_bias,
         transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias,
         transform_3_conv1_weight, transform_3_conv1_bias,
         transform_3_conv2_weight, transform_3_conv2_bias),
    )
    if reverse:
        transforms = transforms[::-1]

    sign = -1.0 if reverse else 1.0
    for w0, b0, w1, b1, w2, b2 in transforms:
        x0 = x[:, :96, :]
        x1 = x[:, 96:, :]
        h = F.relu(F.conv1d(x0, w0, b0, padding=2))
        h = F.relu(F.conv1d(h, w1, b1, padding=2))
        h = F.conv1d(h, w2, b2, padding=2)
        h = h * x_mask
        x = torch.cat((x0, x1 + sign * h), dim=1) * x_mask
    return x
