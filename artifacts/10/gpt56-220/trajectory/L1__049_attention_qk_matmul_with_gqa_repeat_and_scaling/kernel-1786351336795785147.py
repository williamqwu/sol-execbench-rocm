import torch


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    key_states = key[:, :, None].expand(
        batch, kv_heads, heads // kv_heads, slen, head_dim
    ).reshape(batch, heads, slen, head_dim)
    # BF16 operands select the CDNA matrix engines and avoid two FP32 casts.
    return (torch.matmul(query, key_states.transpose(2, 3)) * scaling).to(query.dtype)
