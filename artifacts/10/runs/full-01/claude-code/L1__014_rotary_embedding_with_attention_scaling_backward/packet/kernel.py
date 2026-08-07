import torch
import triton
import triton.language as tl


# The reference computes, per (batch, seq, j):
#   gcf = f32(grad_cos[..., j]) + f32(grad_cos[..., j+HALF])
#   gsf = f32(grad_sin[..., j]) + f32(grad_sin[..., j+HALF])
#   ge  = (gcf * (-sin(emb_j)) * scaling) + (gsf * cos(emb_j) * scaling)
# then reduces over j with a batched GEMV (rocBLAS).
#
# Two numerics traps, both of which cost bit-exactness against the reference:
#
#  1. The compiler happily contracts `t1 + t2` into an FMA with one of the
#     products feeding it, which changes the rounding of the final add. The
#     reference rounds t1 and t2 to f32 separately and then adds. We force a
#     real v_add_f32 with inline asm to reproduce that.
#
#  2. The GEMV's reduction order over j is rocBLAS-internal and not reproducible
#     by any obvious Triton tree/sequential order (measured: only ~25-30% of
#     elements match bit-for-bit, and only ~93% land inside the workload
#     tolerance -- below the 0.99 required ratio). So we keep the same bmm the
#     reference uses, on an identically-laid-out operand, and fuse only the
#     elementwise chain. That is where the memory traffic is anyway: the
#     elementwise pass reads 1024 B/row and the bmm only 256 B/row.


@triton.jit
def _grad_emb_kernel(
    GC, GS, EMB, GE,
    NROW, scaling,
    HD: tl.constexpr,      # head_dim
    HALF: tl.constexpr,    # head_dim // 2
    BLOCK_S: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs < NROW
    offs_j = tl.arange(0, HALF)

    row = offs.to(tl.int64)
    g_off = row[:, None] * HD + offs_j[None, :]
    m = mask_s[:, None]

    gcf = (tl.load(GC + g_off, mask=m, other=0.0).to(tl.float32)
           + tl.load(GC + g_off + HALF, mask=m, other=0.0).to(tl.float32))
    gsf = (tl.load(GS + g_off, mask=m, other=0.0).to(tl.float32)
           + tl.load(GS + g_off + HALF, mask=m, other=0.0).to(tl.float32))

    e = tl.load(EMB + g_off, mask=m, other=0.0)

    t1 = gcf * (-tl.sin(e)) * scaling
    t2 = gsf * tl.cos(e) * scaling

    # Explicit non-contracted f32 add (see note 1 above).
    r = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v",
        [t1, t2], dtype=tl.float32, is_pure=True, pack=1,
    )

    tl.store(GE + row[:, None] * HALF + offs_j[None, :], r, mask=m)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    B, S, HD = emb.shape
    HALF = HD // 2

    if not grad_cos.is_contiguous():
        grad_cos = grad_cos.contiguous()
    if not grad_sin.is_contiguous():
        grad_sin = grad_sin.contiguous()
    if not emb.is_contiguous():
        emb = emb.contiguous()

    grad_emb = torch.empty((B, S, HALF), device=emb.device, dtype=torch.float32)

    rows = B * S
    if rows >= 262144:
        BLOCK_S, num_warps = 4, 2
    else:
        BLOCK_S, num_warps = 4, 1

    _grad_emb_kernel[(triton.cdiv(rows, BLOCK_S),)](
        grad_cos, grad_sin, emb, grad_emb,
        rows, float(attention_scaling),
        HD=HD, HALF=HALF, BLOCK_S=BLOCK_S,
        num_warps=num_warps, num_stages=1,
    )

    # Same operand layout as the reference: transpose(1,2) of a contiguous
    # (B, S, HALF) f32 tensor, contracted against inv_freq_expanded^T.
    out = inv_freq_expanded.transpose(-2, -1) @ grad_emb.transpose(1, 2)
    return out.squeeze(1)
