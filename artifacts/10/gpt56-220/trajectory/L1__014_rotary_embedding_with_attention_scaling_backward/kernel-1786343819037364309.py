import torch
import triton
import triton.language as tl


@triton.jit
def _convert_and_sum(gc, gs, gc_out, gs_out):
    row = tl.program_id(0)
    col = tl.arange(0, 64)
    src = row * 128 + col
    dst = row * 64 + col
    a = tl.load(gc + src).to(tl.float32)
    b = tl.load(gc + src + 64).to(tl.float32)
    c = tl.load(gs + src).to(tl.float32)
    d = tl.load(gs + src + 64).to(tl.float32)
    tl.store(gc_out + dst, a + b)
    tl.store(gs_out + dst, c + d)


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
    _convert_and_sum[(batch * seq_len,)](
        grad_cos, grad_sin, grad_cos_freqs, grad_sin_freqs,
        num_warps=1,
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
