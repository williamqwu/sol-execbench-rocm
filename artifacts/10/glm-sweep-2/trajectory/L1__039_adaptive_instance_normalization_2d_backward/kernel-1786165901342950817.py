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

    x_c = x - mean  # (N,C,H,W)

    # Per-channel reductions over spatial dims -> (N,C)
    s_g = grad_output.sum(dim=(2, 3))
    s_gxc = (grad_output * x_c).sum(dim=(2, 3))
    s_xc = x_c.sum(dim=(2, 3))

    w = weight.view(C)            # (C,)
    s = std.view(N, C)            # (N,C)

    # grad_var (per n,c): w * s_gxc * (-0.5) * std^(-3)
    grad_var = w * s_gxc * (-0.5) * torch.pow(s, -3)          # (N,C)
    # grad_mean (per n,c): -w*s_g/std + grad_var * (-2.0) * s_xc / M
    grad_mean = -w * s_g / s + grad_var * (-2.0) * s_xc / M   # (N,C)

    a = (w / s)                      # (N,C)
    b = (2.0 * grad_var / M)         # (N,C)
    c = (grad_mean / M)             # (N,C)

    a = a.view(N, C, 1, 1)
    b = b.view(N, C, 1, 1)
    c = c.view(N, C, 1, 1)

    grad_input = a * grad_output + b * x_c + c

    grad_bias = s_g.sum(dim=0)                       # (C,)
    grad_weight = (s_gxc / s).sum(dim=0)             # (C,)

    return grad_input, grad_weight, grad_bias
