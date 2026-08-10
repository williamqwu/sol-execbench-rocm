import torch

def _impl(
    v: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Hyena gating and output projection.
    
    Args:
        v: (batch, d_model, L) - filtered output
        x0: (batch, d_model, L) - first gating input
        x1: (batch, d_model, L) - second gating input (for order=2)
        k: (order-1, d_model, L) - filter coefficients
        bias: (order-1, d_model) - bias terms for FFT conv
        out_proj_weight: (d_model, d_model) - output projection weight
        out_proj_bias: (d_model,) - output projection bias
    
    Returns:
        (batch, L, d_model) - gated and projected output
    """
    seqlen = v.shape[-1]
    fft_size = 2 * seqlen
    
    # For order=2, we have one gating step with x1
    # Step 1: Element-wise gating multiplication (dropout=0.0, so no effect)
    v = v * x1
    
    # Step 2: FFT convolution with filter k[0] and bias[0]
    # fftconv_kernel implementation
    k_0 = k[0]  # (d_model, L)
    bias_0 = bias[0]  # (d_model,)
    
    # FFT of filter
    k_f = torch.fft.rfft(k_0, n=fft_size) / fft_size
    # FFT of input
    u_f = torch.fft.rfft(v, n=fft_size)
    
    # Element-wise multiplication in frequency domain and inverse FFT
    # k_f shape: (d_model, fft_size//2+1)
    # u_f shape: (batch, d_model, fft_size//2+1)
    # Need to broadcast k_f
    u_f.mul_(k_f.unsqueeze(0))
    y = torch.fft.irfft(u_f, n=fft_size, norm='forward')[..., :seqlen]
    
    # Add skip connection with bias
    v = torch.addcmul(y, v, bias_0.unsqueeze(-1))
    
    # Step 3: Final gating with x0
    y = v * x0
    
    # Step 4: Transpose from (batch, d_model, L) to (batch, L, d_model)
    y = y.transpose(1, 2)
    
    # Step 5: Output projection: y @ weight.T + bias
    y = torch.matmul(y, out_proj_weight.t()) + out_proj_bias
    
    return y


_compiled_impl = torch.compile(_impl, dynamic=True)


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
    impl = _impl if v.shape[-1] < 700 else _compiled_impl
    return impl(v, x0, x1, k, bias, out_proj_weight, out_proj_bias)
