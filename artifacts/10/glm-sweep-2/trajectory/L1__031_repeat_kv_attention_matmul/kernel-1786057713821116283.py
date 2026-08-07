import torch

@torch.no_grad()
@torch.compile
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    head_dim = 128
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    return attn_weights.to(torch.bfloat16)
