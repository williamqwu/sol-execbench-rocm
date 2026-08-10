import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _project(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias)
    mean, logs = stats.split(weight.shape[0] // 2, dim=1)
    return mean, logs


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return _project(x, weight, bias)
