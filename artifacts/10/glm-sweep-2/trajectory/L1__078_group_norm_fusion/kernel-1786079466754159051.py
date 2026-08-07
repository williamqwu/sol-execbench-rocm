import torch

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    B, C, H, W = x.shape
    num_groups = 32
    x_grouped = x.view(B, num_groups, C // num_groups, H, W).to(torch.float32)
    mean = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
    mean_sq = (x_grouped * x_grouped).mean(dim=[2, 3, 4], keepdim=True)
    var = mean_sq - mean * mean
    x_normalized = (x_grouped - mean) / torch.sqrt(var + eps)
    x_normalized = x_normalized.view(B, C, H, W)
    output = x_normalized * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)
    return output.to(x.dtype)
