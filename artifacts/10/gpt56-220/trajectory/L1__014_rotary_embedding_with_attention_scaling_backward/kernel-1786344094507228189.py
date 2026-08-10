import torch
import triton
import triton.language as tl


@triton.jit
def _convert_and_sum(gc, gs, gc_out, gs_out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    row = offsets // 64
    col = offsets - row * 64
    base = row * 128 + col
    a = tl.load(gc + base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(gc + base + 64, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(gs + base, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(gs + base + 64, mask=mask, other=0.0).to(tl.float32)
    tl.store(gc_out + offsets, a + b, mask=mask)
    tl.store(gs_out + offsets, c + d, mask=mask)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    half_dim = emb.shape[-1] // 2
    batch, seq_len, _ = emb.shape
    grad_cos_freqs = torch.empty(
        (batch, seq_len, half_dim), device=emb.device, dtype=torch.float32
    )
    grad_sin_freqs = torch.empty_like(grad_cos_freqs)
    n = batch * seq_len * half_dim
    _convert_and_sum[(triton.cdiv(n, 2048),)](
        grad_cos, grad_sin, grad_cos_freqs, grad_sin_freqs, n=n, BLOCK=2048,
        num_warps=8,
    )

    emb_half = emb[..., :half_dim]
    grad_emb = (
        grad_cos_freqs * (-emb_half.sin()) * attention_scaling
        + grad_sin_freqs * emb_half.cos() * attention_scaling
    )

    grad_position_ids_expanded = (
        inv_freq_expanded.transpose(-2, -1) @ grad_emb.transpose(1, 2)
    )
    return grad_position_ids_expanded.squeeze(1)
