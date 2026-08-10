import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _project(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    out_channels = weight.shape[0] // 2
    mean = F.conv1d(x, weight[:out_channels], bias[:out_channels])
    logs = F.conv1d(x, weight[out_channels:], bias[out_channels:])
    return mean, logs


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return _project(x, weight, bias)
