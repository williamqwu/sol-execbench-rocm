import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    stats = F.conv1d(x, weight, bias, padding=0)
    return stats[:, :192, :], stats[:, 192:, :]
