import torch
import triton
import triton.language as tl


@triton.jit
def _scale_split(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    two = tl.arange(0, 2)
    p = SRC + offs[:, None] * 2 + two[None, :]
    v = tl.load(p, mask=mask[:, None], other=0.0)
    re, im = tl.split(v)
    tl.store(RE + offs, re / INV, mask=mask)
    tl.store(IM + offs, im / INV, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen

    xf = torch.fft.rfft(x.to(torch.float32), n=fft_size)

    flat = xf.view(torch.float32).reshape(-1)
    M = flat.numel() // 2
    re = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
    im = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

    BLOCK = 1024
    grid = (triton.cdiv(M, BLOCK),)
    _scale_split[grid](flat, re.reshape(-1), im.reshape(-1), M,
                       float(fft_size), BLOCK=BLOCK, num_warps=4)
    return re, im
