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

    x_c = x - mean                              # (N,C,H,W)
    x_norm = x_c / std                           # (N,C,H,W)

    # grad_bias, grad_weight: single-pass reductions matching reference
    grad_bias = grad_output.sum(dim=(0, 2, 3))           # (C,)
    grad_weight = (grad_output * x_norm).sum(dim=(0, 2, 3))  # (C,)

    # Per-(N,C) reductions for grad_input (affine form)
    s_g = grad_output.sum(dim=(2, 3))                    # (N,C)
    s_gxc = (grad_output * x_c).sum(dim=(2, 3))          # (N,C)

    w = weight.view(C)
    s = std.view(N, C)

    grad_var = w * s_gxc * (-0.5) * torch.pow(s, -3)     # (N,C)
    grad_mean = -w * s_g / s + grad_var * (-2.0 * x_c.sum(dim=(2, 3))) / M  # (N,C)

    a = (w / s).view(N, C, 1, 1)
    b = (2.0 * grad_var / M).view(N, C, 1, 1)
    c = (grad_mean / M).view(N, C, 1, 1)

    grad_input = a * grad_output + b * x_c + c

    return grad_input, grad_weight, grad_bias
