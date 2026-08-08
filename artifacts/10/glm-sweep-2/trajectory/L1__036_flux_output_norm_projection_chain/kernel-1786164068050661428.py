import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    inner_dim = temb.shape[-1]
    B, S, D = hidden_states.shape
    N = B * S

    # Flatten to [N, D] for the GEMM-friendly path
    hs = hidden_states.reshape(N, D)

    # LayerNorm without affine (manual, fused-friendly)
    mean = hs.mean(dim=-1, keepdim=True)
    var = hs.var(dim=-1, keepdim=True, unbiased=False)
    hs_norm = (hs - mean) * torch.rsqrt(var + eps)

    # SiLU on temb then modulation GEMM
    temb_silu = F.silu(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    shift = modulation[:, :inner_dim].unsqueeze(1)  # [B,1,D]
    scale = modulation[:, inner_dim:].unsqueeze(1)  # [B,1,D]

    # Modulation: broadcast over seq. Reshape hs_norm to [B,S,D]
    hs_mod = hs_norm.reshape(B, S, D) * (1.0 + scale) + shift  # [B,S,D]

    # Output projection
    output = F.linear(hs_mod, proj_out_weight, proj_out_bias)
    return output
