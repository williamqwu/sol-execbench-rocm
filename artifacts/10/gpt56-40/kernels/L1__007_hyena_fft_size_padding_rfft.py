import torch

@torch.compile(fullgraph=True)
def _compiled_rfft(x: torch.Tensor, seqlen: int):
    x_freq = torch.fft.rfft(x, n=2 * seqlen, norm="forward")
    return x_freq.real, x_freq.imag

@torch.no_grad()
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
    return _compiled_rfft(x_f32, seqlen)
