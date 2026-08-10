import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rope_pointwise(
    grad_cos, grad_sin, emb, out,
    scaling: tl.constexpr, n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    base = pid * 128
    mask = k < 64

    x = tl.load(emb + base + k, mask=mask, other=0.0)
    gc = (tl.load(grad_cos + base + k, mask=mask, other=0.0).to(tl.float32) +
          tl.load(grad_cos + base + 64 + k, mask=mask, other=0.0).to(tl.float32))
    gs = (tl.load(grad_sin + base + k, mask=mask, other=0.0).to(tl.float32) +
          tl.load(grad_sin + base + 64 + k, mask=mask, other=0.0).to(tl.float32))
    val = (gs * libdevice.cos(x) - gc * libdevice.sin(x)) * scaling
    tl.store(out + pid * 64 + k, val, mask=mask)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    batch, n_seq, _ = emb.shape
    grad_emb = torch.empty((batch, n_seq, 64), device=emb.device, dtype=torch.float32)
    _rope_pointwise[(batch * n_seq,)](
        grad_cos, grad_sin, emb, grad_emb,
        scaling=attention_scaling, n_elements=batch * n_seq, BLOCK=64,
        num_warps=1,
    )
    return torch.bmm(inv_freq_expanded.transpose(1, 2), grad_emb.transpose(1, 2)).squeeze(1)
