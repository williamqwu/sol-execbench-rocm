import torch

_EXPS = torch.arange(0, 128, 2, dtype=torch.float32, device='cuda') / 128.0

@torch.compile
def _compute(rope_theta_t, exps):
    return torch.reciprocal(torch.pow(rope_theta_t, exps))

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    rope_theta_t = torch.tensor(float(rope_theta), dtype=torch.float32, device='cuda')
    return _compute(rope_theta_t, _EXPS)
