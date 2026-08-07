import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Fused Conv1d projection and split (kernel_size=1 => GEMM).
    """
    # weight: [out_channels*2, in_channels, 1] -> W2 [out_channels*2, in_channels]
    W2 = weight.squeeze(-1)  # [oc2, ic]
    # x: [B, ic, T] -> stats: [B, oc2, T] = W2 @ x   (broadcasting bias)
    stats = torch.matmul(W2, x)  # [B, oc2, T]
    if bias is not None:
        stats = stats + bias.view(1, -1, 1)
    out_channels = W2.shape[0] // 2
    mean, logs = torch.split(stats, out_channels, dim=1)
    return mean, logs
