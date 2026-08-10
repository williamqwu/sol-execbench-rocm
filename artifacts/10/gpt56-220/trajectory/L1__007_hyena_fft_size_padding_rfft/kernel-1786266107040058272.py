import torch

@torch.no_grad()
@torch.compile(fullgraph=True)
def run(x: torch.Tensor):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    
    Args:
        x: Input tensor of shape (batch, channels, seqlen)
        
    Returns:
        x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
        x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
    """
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    
    # Cast to float32 for FFT numerical stability (already float32 but explicit)
    x_f32 = x.to(torch.float32)
    
    # Zero-pad implicitly and normalize inside rocFFT.
    # Output shape: (batch, channels, seqlen+1) complex
    # Strided component views avoid two full output copies.
    x_freq = torch.fft.rfft(x_f32, n=fft_size, norm="forward")
    return x_freq.real, x_freq.imag
