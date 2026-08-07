import torch

head_dim = 128
_neg_exp = -(torch.arange(0, head_dim, 2, dtype=torch.float32, device='cuda') / float(head_dim))

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return torch.pow(float(rope_theta), _neg_exp)
