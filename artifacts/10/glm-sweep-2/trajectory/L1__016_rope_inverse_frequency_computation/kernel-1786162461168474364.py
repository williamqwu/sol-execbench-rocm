import torch

_EXPS = torch.arange(0, 128, 2, dtype=torch.float32, device='cuda') / 128.0

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return 1.0 / torch.pow(float(rope_theta), _EXPS)
