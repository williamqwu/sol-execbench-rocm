import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # bf16 GEMM with fp32 accumulation (native MFMA on gfx950).
    # Fold scaling into the query: (s*Q) @ K^T == s*(Q@K^T).
    # Pre-scale query in fp32 then cast back to bf16 for the GEMM.
    q = (query.to(torch.float32) * scaling).to(query.dtype)
    k = key
    # Q: [B, 4, S, 256], K^T: [B, 1, 256, S] -> broadcast -> [B, 4, S, S]
    attn = torch.matmul(q, k.transpose(-1, -2))
    return attn.to(query.dtype)
