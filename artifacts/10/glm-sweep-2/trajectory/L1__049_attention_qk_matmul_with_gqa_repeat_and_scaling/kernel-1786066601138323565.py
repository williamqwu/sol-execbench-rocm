import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = key.shape
    # query: [B, 4, S, 256] bf16, key: [B, 1, S, 256] bf16
    # Q @ K^T with broadcasting on head dim (4 vs 1 -> 4). Avoids key repeat copy.
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn * scaling
    return attn.to(query.dtype)
