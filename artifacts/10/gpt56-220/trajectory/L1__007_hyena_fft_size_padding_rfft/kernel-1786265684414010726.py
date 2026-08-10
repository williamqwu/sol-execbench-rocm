import torch

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
    
    # Materialize the zero-padded signal and normalize inside rocFFT.
    # Output shape: (batch, channels, seqlen+1) complex
    x_padded = torch.nn.functional.pad(x_f32, (0, seqlen))
    x_freq = torch.fft.rfft(x_padded, norm="forward")
    
    # Strided component views avoid two full output copies.
    x_freq_real = x_freq.real
    x_freq_imag = x_freq.imag
    
    return x_freq_real, x_freq_imag
