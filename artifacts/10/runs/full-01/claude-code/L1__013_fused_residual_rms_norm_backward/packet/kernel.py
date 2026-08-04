import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused residual RMSNorm backward.
#
# Reference semantics (all math in fp32):
#   go32    = grad_output.float()
#   grad_w  = sum_rows(go32 * normalized)                      -> [N] fp32
#   gn      = go32 * weight
#   m       = mean_over_hidden(gn * normalized)                -> [M,1] fp32
#   grad_x  = rstd * (gn - m * normalized)                     -> [M,N] fp32
#   out     = grad_x.bfloat16()  (returned twice, as two tensors)
#
# Note: `x` is never read by the reference -- it is a saved tensor that the
# backward formula does not need (rstd and `normalized` already carry all the
# information).  So we never touch it, which removes 4 bytes/element of the
# 14 bytes/element the naive dataflow would move.
#
# Per element traffic of this kernel:
#     read  grad_output (bf16)  2 B
#     read  normalized  (fp32)  4 B
#     write grad_hidden (bf16)  2 B
#     write grad_resid  (bf16)  2 B
#                              ----
#                              10 B/elt   (plus O(N) for weight/rstd/grad_w)
#
# One kernel launch does all of it in a single pass: each program owns whole
# rows (so the hidden-dim mean is a purely intra-workgroup reduction), keeps
# `normalized` and `grad_output` live in registers across all three uses, and
# accumulates its private grad_weight partial in registers, flushing it once
# with a single vector atomic add at the end.
# ---------------------------------------------------------------------------


@triton.jit
def _rms_bwd_fused(
    GO,            # *bf16  [M, N]
    NORM,          # *fp32  [M, N]
    RSTD,          # *fp32  [M]
    W,             # *fp32  [N]
    GH,            # *bf16  [M, N]  out
    GR,            # *bf16  [M, N]  out
    GW,            # *fp32  [N]     out (pre-zeroed, atomically accumulated)
    M,             # int32  number of rows
    NPROG,         # int32  grid size (grid-stride step)
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # weight is row-invariant: load once, keep in registers for the whole loop
    w = tl.load(W + cols, mask=mask, other=0.0)

    # private grad_weight partial
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    row = pid
    while row < M:
        off = row.to(tl.int64) * N + cols

        go = tl.load(GO + off, mask=mask, other=0.0).to(tl.float32)
        nm = tl.load(NORM + off, mask=mask, other=0.0)
        rs = tl.load(RSTD + row)

        # grad_weight partial: sum over rows of (go32 * normalized)
        acc += go * nm

        # grad wrt the normalized activations
        gn = go * w

        # mean over the hidden dim of (grad_normalized * normalized)
        s = tl.sum(gn * nm, axis=0) / N

        gx = rs * (gn - s * nm)
        out = gx.to(tl.bfloat16)

        tl.store(GH + off, out, mask=mask)
        tl.store(GR + off, out, mask=mask)

        row += NPROG

    tl.atomic_add(GW + cols, acc, mask=mask)


# Grid sizing: enough workgroups to saturate 256 CUs without generating an
# excessive number of grad_weight atomics (one vector atomic per program).
_MAX_PROG = 2048


def run(grad_output: torch.Tensor,
        x: torch.Tensor,
        normalized: torch.Tensor,
        rstd: torch.Tensor,
        weight: torch.Tensor):
    N = grad_output.shape[-1]
    M = grad_output.numel() // N

    go = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    nm = normalized if normalized.is_contiguous() else normalized.contiguous()
    rs = rstd if rstd.is_contiguous() else rstd.contiguous()
    wt = weight if weight.is_contiguous() else weight.contiguous()

    gh = torch.empty_like(go)
    gr = torch.empty_like(go)
    gw = torch.zeros(N, device=go.device, dtype=torch.float32)

    if M == 0:
        return gh, gr, gw

    BLOCK_N = triton.next_power_of_2(N)
    nprog = M if M < _MAX_PROG else _MAX_PROG

    if BLOCK_N <= 1024:
        num_warps = 4
    elif BLOCK_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    _rms_bwd_fused[(nprog,)](
        go, nm, rs, wt,
        gh, gr, gw,
        M, nprog,
        N=N,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )

    return gh, gr, gw
