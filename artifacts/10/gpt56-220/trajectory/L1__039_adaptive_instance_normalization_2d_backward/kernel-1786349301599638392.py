import torch

torch._dynamo.config.cache_size_limit = 64


@torch.compile(dynamic=False)
def _finish(grad_output, x_centered, weight, std, grad_var, grad_mean, spatial_size: int):
    c = grad_output.shape[1]
    grad_output_scaled = grad_output * weight.view(1, c, 1, 1)
    grad_input = grad_output_scaled / std
    grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
    grad_input = grad_input + grad_mean / spatial_size
    return grad_input


@torch.compile(dynamic=False)
def _prepare(grad_output, x, weight, mean, std):
    c = x.shape[1]
    x_centered = x - mean
    x_normalized = x_centered / std
    grad_weight_terms = grad_output * x_normalized
    grad_output_scaled = grad_output * weight.view(1, c, 1, 1)
    grad_var_terms = grad_output_scaled * x_centered
    grad_mean_terms = grad_output_scaled / (-std)
    return x_centered, grad_weight_terms, grad_var_terms, grad_mean_terms


@torch.compile(dynamic=False)
def _stats(grad_var_sum, grad_mean_sum, centered_sum, std, spatial_size: int):
    grad_var = grad_var_sum * (-0.5) * torch.pow(std, -3)
    grad_mean = grad_mean_sum
    grad_mean = grad_mean + grad_var * (-2.0 * centered_sum) / spatial_size
    return grad_var, grad_mean


@torch.no_grad()
def run(grad_output, x, weight, mean, std):
    _, c, h, w = x.shape
    spatial_size = h * w
    x_centered, grad_weight_terms, grad_var_terms, grad_mean_terms = _prepare(grad_output, x, weight, mean, std)
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = grad_weight_terms.sum(dim=(0, 2, 3))
    grad_var_sum = grad_var_terms.sum(dim=(2, 3), keepdim=True)
    grad_mean_sum = grad_mean_terms.sum(dim=(2, 3), keepdim=True)
    centered_sum = x_centered.sum(dim=(2, 3), keepdim=True)
    grad_var, grad_mean = _stats(grad_var_sum, grad_mean_sum, centered_sum, std, spatial_size)
    grad_input = _finish(grad_output, x_centered, weight, std, grad_var, grad_mean, spatial_size)
    return grad_input, grad_weight, grad_bias
