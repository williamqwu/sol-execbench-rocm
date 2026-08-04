import torch
import aiter


K_DIM = 16384
N_DIM = 7168


@torch.no_grad()
def run(hidden_states, weight, scale_x, scale_w):
    batch_size, seq_len, _ = hidden_states.shape
    m = batch_size * seq_len
    x = hidden_states.reshape(m, K_DIM)
    sx = scale_x.reshape(m, 128)
    sw = scale_w.T.contiguous()
    out = aiter.gemm_a8w8_blockscale(x, weight, sx, sw)
    return out.reshape(batch_size, seq_len, N_DIM)
