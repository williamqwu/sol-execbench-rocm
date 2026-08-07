import torch
import triton
import triton.language as tl


@triton.jit
def _msdb_kernel(
    GO, PA, MSK, DM, OUT,
    H, T, TT,
    keep,
    HAS_DROP: tl.constexpr,
    ACC64: tl.constexpr,
    RCP_HOST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H

    base = pid_bh.to(tl.int64) * TT
    mbase = b.to(tl.int64) * TT

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rm = rows < T
    cm = cols < T
    m2 = rm[:, None] & cm[None, :]

    off = rows[:, None].to(tl.int64) * T + cols[None, :]

    go = tl.load(GO + base + off, mask=m2, other=0.0)
    pa = tl.load(PA + base + off, mask=m2, other=0.0)

    if HAS_DROP:
        dm = tl.load(DM + base + off, mask=m2, other=0)
        # torch lowers `t * dm.float() / (1-p)` to a multiply by the fp32
        # reciprocal of (1-p); reproduce that exactly (a true divide differs
        # in the last bit for ~25% of elements).
        if RCP_HOST:
            g = go * (dm != 0).to(tl.float32) * keep
        else:
            g = go * (dm != 0).to(tl.float32) * (1.0 / keep)
    else:
        g = go

    prod = pa * g
    if ACC64:
        s = tl.sum(prod.to(tl.float64), 1).to(tl.float32)
    else:
        s = tl.sum(prod, 1)

    res = pa * (g - s[:, None])

    mk = tl.load(MSK + mbase + off, mask=m2, other=0)
    res = tl.where(mk != 0, res, 0.0)

    tl.store(OUT + base + off, res, mask=m2)


# (BLOCK_M, num_warps, num_stages) chosen per next_pow2(seq_len).
_TUNE = {
    128: (8, 4, 1),
    256: (4, 4, 1),
    512: (2, 4, 1),
    1024: (1, 4, 1),
    2048: (1, 4, 1),
    4096: (1, 8, 1),
}
_ACC64 = True
_RCP_HOST = False
_OVERRIDE = None


def _cfg(BN):
    if _OVERRIDE is not None:
        return _OVERRIDE
    return _TUNE.get(BN, (1, 8, 1))


def run(grad_output, p_attn, mask, dropout_mask, p_dropout):
    B, H, T, _ = p_attn.shape
    out = torch.empty_like(p_attn)

    p = float(p_dropout.item()) if isinstance(p_dropout, torch.Tensor) else float(p_dropout)
    has_drop = p > 0.0

    # Pass (1-p); the kernel takes its reciprocal in fp32. torch's
    # `x / (1-p)` matches a device-side fp32 reciprocal-multiply bitwise,
    # and for p=0.15 that differs by 1 ulp from a host double reciprocal.
    keep = float(1.0 - p)

    BN = triton.next_power_of_2(T)
    BM, nw, ns = _cfg(BN)
    grid = (triton.cdiv(T, BM), B * H)

    _msdb_kernel[grid](
        grad_output, p_attn,
        mask.view(torch.uint8), dropout_mask.view(torch.uint8), out,
        H, T, T * T,
        keep,
        HAS_DROP=has_drop,
        ACC64=_ACC64,
        RCP_HOST=_RCP_HOST,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_warps=nw,
        num_stages=ns,
    )
    return out
