import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True, dynamic=True)
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias)
    mean, logs = stats.chunk(2, dim=1)
    return mean, logs
