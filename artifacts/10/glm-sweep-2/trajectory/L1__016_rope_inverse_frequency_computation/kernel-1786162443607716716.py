import torch

_ARANGE = torch.arange(0, 64, dtype=torch.float32, device='cuda')
_NEG_EXPS = -_ARANGE / 128.0

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return torch.pow(float(rope_theta), _NEG_EXPS)
