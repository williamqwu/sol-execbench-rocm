import torch
import torch.nn.functional as F


@torch.inference_mode()
def run(
    x,
    input_conv_weight,
    input_conv_bias,
    down1_gn_weight,
    down1_gn_bias,
    down1_conv_weight,
    down1_conv_bias,
    down1_res_weight,
    down1_res_bias,
    down2_gn_weight,
    down2_gn_bias,
    down2_conv_weight,
    down2_conv_bias,
    down2_res_weight,
    down2_res_bias,
    down3_gn_weight,
    down3_gn_bias,
    down3_conv_weight,
    down3_conv_bias,
    down3_res_weight,
    down3_res_bias,
    latent_gn_weight,
    latent_gn_bias,
    latent_conv_weight,
    latent_conv_bias,
    eps,
):
    h = F.conv3d(x, input_conv_weight, input_conv_bias, stride=1, padding=1)

    identity = F.conv3d(h, down1_res_weight, down1_res_bias, stride=2)
    h = F.group_norm(h, 8, down1_gn_weight, down1_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down1_conv_weight, down1_conv_bias, stride=2, padding=1)
    h = h + identity

    identity = F.conv3d(h, down2_res_weight, down2_res_bias, stride=2)
    h = F.group_norm(h, 16, down2_gn_weight, down2_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down2_conv_weight, down2_conv_bias, stride=2, padding=1)
    h = h + identity

    identity = F.conv3d(h, down3_res_weight, down3_res_bias, stride=(1, 2, 2))
    h = F.group_norm(h, 32, down3_gn_weight, down3_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down3_conv_weight, down3_conv_bias, stride=(1, 2, 2), padding=1)
    h = h + identity

    h = F.group_norm(h, 32, latent_gn_weight, latent_gn_bias, eps)
    h = F.silu(h)
    return F.conv3d(h, latent_conv_weight, latent_conv_bias, stride=1, padding=1)
