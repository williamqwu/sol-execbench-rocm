import torch


@torch.compile(dynamic=True)
def _rope(ids: torch.Tensor, theta: float):
    device = ids.device
    e8 = torch.arange(8, dtype=torch.float32, device=device)
    e28 = torch.arange(28, dtype=torch.float32, device=device)
    b8 = torch.pow(theta, -e8 / 8.0)
    b28 = torch.pow(theta, -e28 / 28.0)
    a0 = ids[:, 0:1] * b8
    a1 = ids[:, 1:2] * b28
    a2 = ids[:, 2:3] * b28
    angles = torch.cat((a0, a1, a2), dim=1)
    return torch.repeat_interleave(torch.cos(angles), 2, dim=1), torch.repeat_interleave(torch.sin(angles), 2, dim=1)


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    return _rope(ids, theta)
