import torch
import numpy as np

_EXPS = np.array([2.0 * i / 128.0 for i in range(64)], dtype=np.float64)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    powers = np.exp(_EXPS * np.log(float(rope_theta)))
    inv = (1.0 / powers).astype(np.float32)
    return torch.from_numpy(inv).to('cuda', non_blocking=True)
