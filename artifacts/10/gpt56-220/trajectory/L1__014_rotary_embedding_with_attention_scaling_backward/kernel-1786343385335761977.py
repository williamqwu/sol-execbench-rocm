import torch
import triton
import triton.language as tl


@triton.jit
def _rope_backward(
    grad_cos, grad_sin, emb, inv_freq, out,
    scaling: tl.constexpr, n_seq: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // n_seq
    pos = pid - batch * n_seq
    k = tl.arange(0, BLOCK)
    base = (batch * n_seq + pos) * 128
    mask = k < 64

    x = tl.load(emb + base + k, mask=mask, other=0.0)
    gc = (tl.load(grad_cos + base + k, mask=mask, other=0.0).to(tl.float32) +
          tl.load(grad_cos + base + 64 + k, mask=mask, other=0.0).to(tl.float32))
    gs = (tl.load(grad_sin + base + k, mask=mask, other=0.0).to(tl.float32) +
          tl.load(grad_sin + base + 64 + k, mask=mask, other=0.0).to(tl.float32))
    inv = tl.load(inv_freq + batch * 64 + k, mask=mask, other=0.0)
    val = (gs * tl.cos(x) - gc * tl.sin(x)) * inv
    tl.store(out + pid, tl.sum(val, axis=0) * scaling)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    batch, n_seq, _ = emb.shape
    out = torch.empty((batch, n_seq), device=emb.device, dtype=torch.float32)
    _rope_backward[(batch * n_seq,)](
        grad_cos, grad_sin, emb, inv_freq_expanded, out,
        scaling=attention_scaling, n_seq=n_seq, BLOCK=64,
        num_warps=1,
    )
    return out
