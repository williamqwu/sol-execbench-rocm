import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    v: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
) -> torch.Tensor:
    seqlen = v.shape[-1]
    fft_size = 2 * seqlen

    # Step 1: gating with x1
    v = v * x1

    # Step 2: FFT convolution with filter k[0] and bias[0]
    k_0 = k[0]
    bias_0 = bias[0]

    # Pre-pad to fft_size so rocFFT uses a direct (non-padding) plan.
    # Bit-identical: zero-padding to n then rfft == rfft(..., n=n).
    v_pad = F.pad(v, (0, fft_size - seqlen))
    k_pad = F.pad(k_0, (0, fft_size - seqlen))

    k_f = torch.fft.rfft(k_pad) / fft_size
    u_f = torch.fft.rfft(v_pad)
    y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=fft_size, norm='forward')[..., :seqlen]

    # Steps 3: skip connection + final gating fused
    y = (y + v * bias_0.unsqueeze(-1)) * x0

    # Step 4: transpose (batch, d_model, L) -> (batch, L, d_model)
    y = y.transpose(1, 2)

    # Step 5: output projection
    y = torch.matmul(y, out_proj_weight.t()) + out_proj_bias

    return y
