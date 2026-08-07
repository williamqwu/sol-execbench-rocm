import torch

_RUN = torch.compile(
    lambda rope_theta, head_dim: 1.0 / torch.pow(float(rope_theta), torch.arange(0, head_dim, 2, dtype=torch.float32, device='cuda') / float(head_dim)),
    fullgraph=False,
)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    head_dim = 128
    return _RUN(rope_theta, head_dim)
