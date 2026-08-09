import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True, dynamic=True)
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias)
    mean = stats.narrow(1, 0, 192)
    logs = stats.narrow(1, 192, 192)
    return mean, logs
