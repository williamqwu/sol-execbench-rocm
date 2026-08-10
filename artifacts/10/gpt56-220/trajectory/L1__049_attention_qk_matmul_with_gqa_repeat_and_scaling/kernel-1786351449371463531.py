import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # Scaling Q is algebraically equivalent, and for the fixed power-of-two
    # scale it is exact in BF16.  It avoids scaling the much larger S x S
    # attention matrix after GEMM. Matmul broadcasts the singleton KV head.
    return torch.matmul(query * scaling, key.transpose(2, 3))
