import torch


@torch.compile(dynamic=True)
def _compiled(grad_output, x, weight, mean, std):
    _, c, h, w = x.shape
    m = h * w
    xc = x - mean
    xn = xc / std
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = (grad_output * xn).sum(dim=(0, 2, 3))
    gos = grad_output * weight.view(1, c, 1, 1)
    grad_var = (gos * xc).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
    grad_mean = (gos / (-std)).sum(dim=(2, 3), keepdim=True)
    grad_mean = grad_mean + grad_var * (-2.0 * xc).sum(dim=(2, 3), keepdim=True) / m
    grad_input = gos / std + grad_var * 2.0 * xc / m + grad_mean / m
    return grad_input, grad_weight, grad_bias


@torch.no_grad()
def run(grad_output, x, weight, mean, std):
    return _compiled(grad_output, x, weight, mean, std)
