import torch


@torch.compile(dynamic=True)
def _finish(grad_output, x_centered, weight, std, grad_var, grad_mean, spatial_size: int):
    c = grad_output.shape[1]
    grad_output_scaled = grad_output * weight.view(1, c, 1, 1)
    grad_input = grad_output_scaled / std
    grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
    grad_input = grad_input + grad_mean / spatial_size
    return grad_input


@torch.no_grad()
def run(grad_output, x, weight, mean, std):
    _, c, h, w = x.shape
    spatial_size = h * w
    x_centered = x - mean
    x_normalized = x_centered / std
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))
    weight_reshaped = weight.view(1, c, 1, 1)
    grad_var = (grad_output * x_centered).sum(dim=(2, 3), keepdim=True) * weight_reshaped * (-0.5) * torch.pow(std, -3)
    grad_mean = grad_output.sum(dim=(2, 3), keepdim=True) * weight_reshaped / (-std)
    grad_mean = grad_mean + grad_var * (-2.0 * x_centered).sum(dim=(2, 3), keepdim=True) / spatial_size
    grad_input = _finish(grad_output, x_centered, weight, std, grad_var, grad_mean, spatial_size)
    return grad_input, grad_weight, grad_bias
