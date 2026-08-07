import torch

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
    # Reference uses fft_size = 2*seqlen, but nearly every seqlen is prime,
    # so 2*seqlen has a huge prime factor and rocFFT falls back to the slow
    # Bluestein (chirp-z) path.  Padding to the next power of two is exact:
    # circular convolution with N >= 2*L-1 equals linear convolution in the
    # first L samples, and N_p2 >= 2*L >= 2*L-1 always holds.
    n = 2 * seqlen
    fft_size = 1 << (n - 1).bit_length()

    # Step 1: gating with x1
    v = v * x1

    # Step 2: FFT convolution with filter k[0] and bias[0]
    k_0 = k[0]
    bias_0 = bias[0]

    k_f = torch.fft.rfft(k_0.to(torch.float32), n=fft_size) / fft_size
    u_f = torch.fft.rfft(v.to(torch.float32), n=fft_size)
    y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=fft_size, norm='forward')[..., :seqlen]

    v = y + v * bias_0.unsqueeze(-1)
    v = v.to(torch.float32)

    # Step 3: final gating with x0
    y = v * x0

    # Step 4: transpose (batch, d_model, L) -> (batch, L, d_model)
    y = y.transpose(1, 2)

    # Step 5: output projection
    y = torch.matmul(y, out_proj_weight.t()) + out_proj_bias

    return y
