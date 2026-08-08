import torch
import math

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    head_dim = 128
    vals = [1.0 / (float(rope_theta) ** (2.0 * i / head_dim)) for i in range(head_dim // 2)]
    return torch.tensor(vals, dtype=torch.float32, device='cuda')
