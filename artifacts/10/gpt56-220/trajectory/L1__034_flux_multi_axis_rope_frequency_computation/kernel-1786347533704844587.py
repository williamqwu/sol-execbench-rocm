import torch


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    device = ids.device
    bands8 = torch.pow(theta, -torch.arange(8, device=device, dtype=torch.float32) / 8.0)
    bands28 = torch.pow(theta, -torch.arange(28, device=device, dtype=torch.float32) / 28.0)
    bands = torch.cat((bands8, bands28, bands28))
    axes = torch.cat((torch.zeros(8, device=device, dtype=torch.long),
                      torch.ones(28, device=device, dtype=torch.long),
                      torch.full((28,), 2, device=device, dtype=torch.long)))
    angles = ids[:, axes] * bands
    return (torch.repeat_interleave(torch.cos(angles), 2, dim=1),
            torch.repeat_interleave(torch.sin(angles), 2, dim=1))
