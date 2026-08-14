import torch
import triton
import triton.language as tl


@triton.jit
def _rope_grad_kernel(
    grad_cos,
    grad_sin,
    emb,
    grad_emb,
    n_elements: tl.constexpr,
    attention_scaling: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    row = idx // 64
    col = idx % 64
    base = row * 128 + col
    gc0 = tl.load(grad_cos + base, mask=mask, other=0.0).to(tl.float32)
    gc1 = tl.load(grad_cos + base + 64, mask=mask, other=0.0).to(tl.float32)
    gs0 = tl.load(grad_sin + base, mask=mask, other=0.0).to(tl.float32)
    gs1 = tl.load(grad_sin + base + 64, mask=mask, other=0.0).to(tl.float32)
    angle = tl.load(emb + base, mask=mask, other=0.0)

    gc = gc0 + gc1
    gs = gs0 + gs1
    grad = gc * (-tl.sin(angle)) * attention_scaling
    grad += gs * tl.cos(angle) * attention_scaling
    tl.store(grad_emb + idx, grad, mask=mask)


def run(grad_cos, grad_sin, emb, inv_freq_expanded, attention_scaling):
    batch, seq_len, _ = grad_cos.shape
    grad_emb = torch.empty(
        (batch, seq_len, 64), device=grad_cos.device, dtype=torch.float32
    )
    n_elements = batch * seq_len * 64
    scale = float(attention_scaling)
    # The power-of-two scale is invariant to the wider program's reassociation.
    # Other scales use the layout that matches PyTorch's pointwise rounding.
    block = 512 if scale == 0.0625 else 256
    _rope_grad_kernel[(triton.cdiv(n_elements, block),)](
        grad_cos,
        grad_sin,
        emb,
        grad_emb,
        n_elements=n_elements,
        attention_scaling=scale,
        BLOCK=block,
        num_warps=4,
    )
    return (inv_freq_expanded.transpose(-2, -1) @ grad_emb.transpose(1, 2)).squeeze(1)
