import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.linear(x.transpose(1, 2), weight[:, :, 0], bias)
    mean, logs = stats.split(weight.shape[0] // 2, dim=2)
    return mean.transpose(1, 2), logs.transpose(1, 2)
