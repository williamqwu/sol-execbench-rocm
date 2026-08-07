import torch
import torch.nn.functional as F

@torch.no_grad()
def _impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias, padding=0)
    out_channels = weight.shape[0] // 2
    mean, logs = torch.split(stats, out_channels, dim=1)
    return mean, logs

_compiled = torch.compile(_impl, dynamic=True)

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return _compiled(x, weight, bias)
