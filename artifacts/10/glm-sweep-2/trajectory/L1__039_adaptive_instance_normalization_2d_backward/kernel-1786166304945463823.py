import torch

_compiled = None

def _fn(grad_output, x, weight, mean, std):
    N, C, H, W = x.shape
    spatial_size = H * W

    x_centered = x - mean
    x_normalized = x_centered / std

    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))

    weight_reshaped = weight.view(1, C, 1, 1)
    grad_output_scaled = grad_output * weight_reshaped

    grad_var = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)

    grad_mean = (grad_output_scaled / (-std)).sum(dim=(2, 3), keepdim=True)
    grad_mean = grad_mean + grad_var * (-2.0 * x_centered).sum(dim=(2, 3), keepdim=True) / spatial_size

    grad_input = grad_output_scaled / std
    grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
    grad_input = grad_input + grad_mean / spatial_size

    return grad_input, grad_weight, grad_bias

_compiled = torch.compile(_fn, mode="max-autotune", dynamic=True)

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    return _compiled(grad_output, x, weight, mean, std)
