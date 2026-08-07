import torch
import torch.nn.functional as F


def _run_compiled(
    x, input_conv_weight, input_conv_bias,
    down1_gn_weight, down1_gn_bias,
    down1_conv_weight, down1_conv_bias,
    down1_res_weight, down1_res_bias,
    down2_gn_weight, down2_gn_bias,
    down2_conv_weight, down2_conv_bias,
    down2_res_weight, down2_res_bias,
    down3_gn_weight, down3_gn_bias,
    down3_conv_weight, down3_conv_bias,
    down3_res_weight, down3_res_bias,
    latent_gn_weight, latent_gn_bias,
    latent_conv_weight, latent_conv_bias,
    eps,
):
    num_groups = 32

    h = F.conv3d(x, input_conv_weight, input_conv_bias, stride=1, padding=1)

    identity1 = F.conv3d(h, down1_res_weight, down1_res_bias, stride=2)
    h = F.group_norm(h, num_groups // 4, down1_gn_weight, down1_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down1_conv_weight, down1_conv_bias, stride=2, padding=1)
    h = h + identity1

    identity2 = F.conv3d(h, down2_res_weight, down2_res_bias, stride=2)
    h = F.group_norm(h, num_groups // 2, down2_gn_weight, down2_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down2_conv_weight, down2_conv_bias, stride=2, padding=1)
    h = h + identity2

    identity3 = F.conv3d(h, down3_res_weight, down3_res_bias, stride=(1, 2, 2))
    h = F.group_norm(h, num_groups, down3_gn_weight, down3_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down3_conv_weight, down3_conv_bias, stride=(1, 2, 2), padding=1)
    h = h + identity3

    h = F.group_norm(h, num_groups, latent_gn_weight, latent_gn_bias, eps)
    h = F.silu(h)
    output = F.conv3d(h, latent_conv_weight, latent_conv_bias, stride=1, padding=1)

    return output


_compiled = torch.compile(_run_compiled, mode="default", dynamic=True)


@torch.no_grad()
def run(
    x: torch.Tensor,
    input_conv_weight: torch.Tensor,
    input_conv_bias: torch.Tensor,
    down1_gn_weight: torch.Tensor,
    down1_gn_bias: torch.Tensor,
    down1_conv_weight: torch.Tensor,
    down1_conv_bias: torch.Tensor,
    down1_res_weight: torch.Tensor,
    down1_res_bias: torch.Tensor,
    down2_gn_weight: torch.Tensor,
    down2_gn_bias: torch.Tensor,
    down2_conv_weight: torch.Tensor,
    down2_conv_bias: torch.Tensor,
    down2_res_weight: torch.Tensor,
    down2_res_bias: torch.Tensor,
    down3_gn_weight: torch.Tensor,
    down3_gn_bias: torch.Tensor,
    down3_conv_weight: torch.Tensor,
    down3_conv_bias: torch.Tensor,
    down3_res_weight: torch.Tensor,
    down3_res_bias: torch.Tensor,
    latent_gn_weight: torch.Tensor,
    latent_gn_bias: torch.Tensor,
    latent_conv_weight: torch.Tensor,
    latent_conv_bias: torch.Tensor,
    eps: float,
):
    return _compiled(
        x, input_conv_weight, input_conv_bias,
        down1_gn_weight, down1_gn_bias,
        down1_conv_weight, down1_conv_bias,
        down1_res_weight, down1_res_bias,
        down2_gn_weight, down2_gn_bias,
        down2_conv_weight, down2_conv_bias,
        down2_res_weight, down2_res_bias,
        down3_gn_weight, down3_gn_bias,
        down3_conv_weight, down3_conv_bias,
        down3_res_weight, down3_res_bias,
        latent_gn_weight, latent_gn_bias,
        latent_conv_weight, latent_conv_bias,
        eps,
    )
