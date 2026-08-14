import torch
import triton
import triton.language as tl


@triton.jit
def _av_kernel(
    A, V, O,
    S, num_m,
    sab, sah, sam,
    svb, svh, svk,
    sob, som,
    H: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    EM: tl.constexpr, EK: tl.constexpr,
):
    pid = tl.program_id(0)
    # m-major ordering: consecutive programs share the same (b, h) and therefore
    # the same V panel, which keeps the re-read of V resident in L2.
    m_id = pid % num_m
    bh = pid // num_m
    h = bh % H
    b = bh // H

    ab = A + b * sab + h * sah
    vb = V + b * svb + h * svh
    ob = O + b * sob + h * D

    om = m_id * BM + tl.arange(0, BM)
    od = tl.arange(0, D)
    ok = tl.arange(0, BK)

    ap = ab + om[:, None] * sam + ok[None, :]
    vp = vb + ok[:, None] * svk + od[None, :]
    mm = om < S

    acc = tl.zeros((BM, D), dtype=tl.float32)
    for k0 in range(0, S, BK):
        if EK:
            if EM:
                a = tl.load(ap)
            else:
                a = tl.load(ap, mask=mm[:, None], other=0.0)
            v = tl.load(vp)
        else:
            km = (k0 + ok) < S
            if EM:
                a = tl.load(ap, mask=km[None, :], other=0.0)
            else:
                a = tl.load(ap, mask=mm[:, None] & km[None, :], other=0.0)
            v = tl.load(vp, mask=km[:, None], other=0.0)
        acc = tl.dot(a, v, acc)
        ap += BK
        vp += BK * svk

    o = acc.to(O.dtype.element_ty)
    op = ob + om[:, None] * som + od[None, :]
    if EM:
        tl.store(op, o)
    else:
        tl.store(op, o, mask=mm[:, None])


# Empirically measured best (BM, BK, num_warps, num_stages) per (batch, seq_len),
# obtained by exhaustive sweep on MI355X. Tile choice trades V re-read traffic
# (V is re-read ceil(S/BM) times per head) against CTA-level parallelism.
_CFG = {
    (32, 128): (128, 32, 4, 3),
    (16, 256): (256, 32, 8, 2),
    (1, 256): (64, 64, 8, 3),
    (64, 128): (128, 32, 4, 3),
    (4, 512): (256, 32, 8, 2),
    (2, 541): (64, 32, 4, 1),
    (1, 293): (64, 64, 8, 2),
    (1, 131): (32, 32, 4, 2),
    (1, 853): (64, 32, 4, 1),
    (8, 512): (256, 64, 8, 2),
    (8, 128): (64, 64, 4, 2),
    (2, 1024): (64, 64, 8, 3),
    (8, 256): (128, 32, 8, 3),
    (1, 2048): (64, 64, 8, 3),
    (4, 691): (256, 64, 8, 1),
    (1, 997): (64, 64, 4, 1),
}

_H = 40


def _heuristic(B, S):
    """Fallback for shapes not in the measured table."""
    aligned = (S % 64 == 0)
    # Largest tile that still fills the 256 CUs, capped by S.
    best = 64
    for bm in (256, 128, 64, 32):
        if bm > S and bm > 32:
            continue
        if B * _H * triton.cdiv(S, bm) >= 256:
            best = bm
            break
    if not aligned and B * _H <= 128:
        # Low CTA count with a ragged tail: small tiles schedule better.
        best = min(best, 64)
    bk = 32 if best >= 128 else 64
    nw = 8 if best >= 128 else 4
    ns = 3 if aligned else 1
    return best, bk, nw, ns


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    B, H, S, _ = attn_weights.shape
    D = value_states.shape[-1]

    cfg = _CFG.get((B, S))
    if cfg is None or H != _H:
        cfg = _heuristic(B, S)
    BM, BK, nw, ns = cfg

    out = torch.empty((B, S, H * D), dtype=attn_weights.dtype,
                      device=attn_weights.device)

    num_m = triton.cdiv(S, BM)
    _av_kernel[(B * H * num_m,)](
        attn_weights, value_states, out,
        S, num_m,
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2),
        value_states.stride(0), value_states.stride(1), value_states.stride(2),
        out.stride(0), out.stride(1),
        H=H, D=D,
        BM=BM, BK=BK,
        EM=(S % BM == 0), EK=(S % BK == 0),
        num_warps=nw, num_stages=ns,
    )
    return out
