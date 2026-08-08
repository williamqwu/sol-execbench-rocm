import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    N, C, H, W = x.shape
    M = H * W

    x_centered = x - mean
    x_normalized = x_centered / std

    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))

    weight_r = weight.view(1, C, 1, 1)
    grad_output_scaled = grad_output * weight_r

    r1 = grad_output_scaled * x_centered
    r2 = grad_output_scaled / (-std)
    r3 = -2.0 * x_centered
    stacked = torch.stack([r1, r2, r3], dim=0)
    sums = stacked.sum(dim=(3, 4))
    s1, s2, s3 = sums[0], sums[1], sums[2]

    s_flat = std.view(N, C)
    grad_var = s1 * (-0.5) * torch.pow(s_flat, -3)
    grad_mean = s2 + grad_var * s3 / M

    grad_input = grad_output_scaled / std + grad_var.view(N, C, 1, 1) * 2.0 * x_centered / M + grad_mean.view(N, C, 1, 1) / M

    return grad_input, grad_weight, grad_bias
