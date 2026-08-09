import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv2d(x.unsqueeze(2), weight.unsqueeze(2), bias)
    mean, logs = stats.split(weight.shape[0] // 2, dim=1)
    return mean.squeeze(2), logs.squeeze(2)
