import math
import numpy as np
import torch

head_dim = 128
_two_over_hd = np.arange(0, head_dim // 2, dtype=np.float32) * (2.0 / float(head_dim))

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    arr = np.power(float(rope_theta), -_two_over_hd, dtype=np.float64).astype(np.float32)
    return torch.from_numpy(arr).to('cuda', non_blocking=True)
