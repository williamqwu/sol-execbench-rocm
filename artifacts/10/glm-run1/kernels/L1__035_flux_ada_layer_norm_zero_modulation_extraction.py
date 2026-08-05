import torch
import torch.nn.functional as F

@torch.no_grad()
def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    AdaLayerNormZero modulation parameter extraction.
    Fused linear (bias-add fused into GEMM) then chunk into 6 modulation params.
    """
    emb_out = F.linear(emb, weight, bias)
    return emb_out.chunk(6, dim=1)
