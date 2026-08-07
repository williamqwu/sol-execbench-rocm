import torch

@torch.no_grad()
def run(x: torch.Tensor):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen

    x_freq = torch.fft.rfft(x, n=fft_size, norm='forward')

    return x_freq.real, x_freq.imag
