import torch

@torch.no_grad()
@torch.compile
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    head_dim = 128
    scaling = head_dim ** -0.5
    # Fold scaling into query (small) so the GEMM writes bf16 directly with no
    # separate epilogue pass over the large [B,H,S,S] output: (q*s)@k^T = s*(q@k^T)
    attn_weights = torch.matmul(query * scaling, key.transpose(2, 3))
    return attn_weights.to(torch.bfloat16)
