import torch
import torch.nn.functional as F


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"coordinate_descent_tuning": True},
)
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias)
    mean, logs = stats.split(weight.shape[0] // 2, dim=1)
    return mean, logs
