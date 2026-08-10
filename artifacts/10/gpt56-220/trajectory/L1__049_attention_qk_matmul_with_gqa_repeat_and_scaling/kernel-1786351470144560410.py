import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    # Scale the single KV head before expanding it.  For the fixed power-of-two
    # scale this is exact in BF16 and touches 4x fewer elements than scaling Q.
    scaled_key = key * scaling
    key_states = scaled_key[:, :, None].expand(
        batch, kv_heads, heads // kv_heads, slen, head_dim
    ).reshape(batch, heads, slen, head_dim)
    return torch.matmul(query, key_states.transpose(2, 3))
