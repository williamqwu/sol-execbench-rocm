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
    spatial_size = H * W

    x_centered = x - mean
    inv_std = 1.0 / std

    # grad_bias = sum over N,H,W of grad_output
    grad_bias = grad_output.sum(dim=(0, 2, 3))

    # grad_weight = sum over N,H,W of grad_output * x_normalized
    grad_weight = (grad_output * x_centered * inv_std).sum(dim=(0, 2, 3))

    weight_reshaped = weight.view(1, C, 1, 1)
    grad_output_scaled = grad_output * weight_reshaped

    # reduction over (H,W) per (N,C)
    go_scaled_xc_sum = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True)
    go_scaled_sum = grad_output_scaled.sum(dim=(2, 3), keepdim=True)
    xc_sum = x_centered.sum(dim=(2, 3), keepdim=True)

    inv_spatial = 1.0 / spatial_size
    inv_std_neg = -inv_std

    grad_var = go_scaled_xc_sum * (-0.5) * (inv_std * inv_std * inv_std)
    grad_mean = go_scaled_sum * inv_std_neg + grad_var * (-2.0) * xc_sum * inv_spatial

    grad_input = grad_output_scaled * inv_std \
        + grad_var * (2.0 * inv_spatial) * x_centered \
        + grad_mean * inv_spatial

    return grad_input, grad_weight, grad_bias
