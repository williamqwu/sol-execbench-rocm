import torch


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # BF16 operands select the CDNA matrix engines; the singleton KV head is
    # broadcast across query heads without materializing repeated keys.
    return (torch.matmul(query, key.transpose(2, 3)) * scaling).to(query.dtype)
