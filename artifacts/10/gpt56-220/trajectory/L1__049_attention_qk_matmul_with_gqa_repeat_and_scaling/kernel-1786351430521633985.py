import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    key_states = key[:, :, None].expand(
        batch, kv_heads, heads // kv_heads, slen, head_dim
    ).reshape(batch, heads, slen, head_dim)

    # Scaling Q is algebraically equivalent, and for the fixed power-of-two
    # scale it is exact in BF16.  It avoids scaling the much larger S x S
    # attention matrix after GEMM.
    return torch.matmul(query * scaling, key_states.transpose(2, 3))
