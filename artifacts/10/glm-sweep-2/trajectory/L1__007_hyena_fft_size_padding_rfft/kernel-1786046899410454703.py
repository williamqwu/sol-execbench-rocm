import torch

@torch.no_grad()
def run(x: torch.Tensor):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    scale = 1.0 / fft_size

    x_freq = torch.fft.rfft(x, n=fft_size)

    x_freq_real = x_freq.real * scale
    x_freq_imag = x_freq.imag * scale

    return x_freq_real, x_freq_imag
