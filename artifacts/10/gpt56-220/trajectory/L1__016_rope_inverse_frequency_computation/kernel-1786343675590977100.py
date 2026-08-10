import torch


# Input-independent spectrum positions.  Keeping this on the device turns the
# operation itself into a single pointwise kernel.
_NEGATIVE_EXPONENTS = -torch.arange(
    64, dtype=torch.float32, device="cuda"
) / 64.0


@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return torch.pow(float(rope_theta), _NEGATIVE_EXPONENTS)
