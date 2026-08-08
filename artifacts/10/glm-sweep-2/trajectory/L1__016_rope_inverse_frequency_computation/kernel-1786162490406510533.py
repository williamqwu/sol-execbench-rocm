import torch

_EXPS = torch.arange(0, 128, 2, dtype=torch.float32, device='cuda') / 128.0

@torch.compile
def _compute(rope_theta, exps):
    return 1.0 / torch.pow(rope_theta, exps)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return _compute(float(rope_theta), _EXPS)
