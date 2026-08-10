import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap_softmax(x, y, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(x + row * n_cols + cols, mask=mask, other=-float("inf")).to(tl.float32)
    v = libdevice.tanh(v * (1.0 / 30.0)) * 30.0
    v = v - tl.max(v, axis=0)
    e = tl.exp(v)
    out = e / tl.sum(e, axis=0)
    tl.store(y + row * n_cols + cols, out, mask=mask)


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    n_cols = attn_weights.shape[-1]
    rows = attn_weights.numel() // n_cols
    out = torch.empty_like(attn_weights)
    block = triton.next_power_of_2(n_cols)
    _softcap_softmax[(rows,)](attn_weights, out, n_cols=n_cols, BLOCK=block,
                              num_warps=2)
    return out
