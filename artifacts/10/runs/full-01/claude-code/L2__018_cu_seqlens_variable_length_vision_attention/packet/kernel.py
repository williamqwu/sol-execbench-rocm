import torch
import torch.nn.functional as F
import triton
import triton.language as tl

EMBED = 1152
NH = 16
HD = 72
QKVD = 3456
HALF = HD // 2
FUSE_N = 1500


@triton.jit
def _rope_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, NTOT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < NTOT
    n = i // 1152
    r = i % 1152
    h = r // 72
    d = r - h * 72
    lo = d < 36
    r2 = tl.where(lo, r + 36, r - 36)

    c = tl.load(cos_ptr + i, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + i, mask=mask, other=0.0).to(tl.float32)
    s = tl.where(lo, -s, s)

    base = n * 3456 + r
    base2 = n * 3456 + r2

    xq = tl.load(qkv_ptr + base, mask=mask, other=0.0).to(tl.float32)
    xq2 = tl.load(qkv_ptr + base2, mask=mask, other=0.0).to(tl.float32)
    t1 = (xq * c).to(tl.bfloat16).to(tl.float32)
    t2 = (xq2 * s).to(tl.bfloat16).to(tl.float32)
    tl.store(q_ptr + i, (t1 + t2).to(tl.bfloat16), mask=mask)

    xk = tl.load(qkv_ptr + base + 1152, mask=mask, other=0.0).to(tl.float32)
    xk2 = tl.load(qkv_ptr + base2 + 1152, mask=mask, other=0.0).to(tl.float32)
    u1 = (xk * c).to(tl.bfloat16).to(tl.float32)
    u2 = (xk2 * s).to(tl.bfloat16).to(tl.float32)
    tl.store(k_ptr + i, (u1 + u2).to(tl.bfloat16), mask=mask)


@triton.jit
def _attn_kernel(
    q_ptr, k_ptr, qkv_ptr, o_ptr, cu_ptr,
    S, SCALE,
    SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)

    offs = tl.arange(0, SP)
    valid = offs < S
    cu = tl.load(cu_ptr + offs, mask=valid, other=0).to(tl.int32)
    cup = tl.load(cu_ptr + offs - 1, mask=valid & (offs > 0), other=0).to(tl.int32)
    lens = tl.where(valid, cu - cup, 0)
    nb = (lens + (BM - 1)) // BM
    cnb = tl.cumsum(nb, axis=0)
    sid = tl.sum((cnb <= pid).to(tl.int32), axis=0)

    if sid < S:
        sel = offs == sid
        start_blk = tl.sum(tl.where(sel, cnb - nb, 0), axis=0)
        seq_start = tl.sum(tl.where(sel, cup, 0), axis=0)
        L = tl.sum(tl.where(sel, lens, 0), axis=0)

        m0 = (pid - start_blk) * BM
        offm = m0 + tl.arange(0, BM)
        mmask = offm < L
        offd = tl.arange(0, BD)
        dmask = offd < 72

        hoff = h * 72
        qrow = (seq_start + offm)[:, None] * 1152 + hoff + offd[None, :]
        q = tl.load(q_ptr + qrow, mask=mmask[:, None] & dmask[None, :], other=0.0)

        m_i = tl.full([BM], -1.0e30, tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, BD], tl.float32)

        nblk = tl.cdiv(L, BN)
        for sn in range(0, nblk):
            offn = sn * BN + tl.arange(0, BN)
            nmask = offn < L
            krow = (seq_start + offn)[:, None] * 1152 + hoff + offd[None, :]
            k = tl.load(k_ptr + krow, mask=nmask[:, None] & dmask[None, :], other=0.0)
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            s = s.to(tl.bfloat16).to(tl.float32) * SCALE
            s = s.to(tl.bfloat16).to(tl.float32)
            s = tl.where(nmask[None, :], s, -1.0e30)
            mn = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - mn)
            p = tl.exp(s - mn[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = mn
            acc = acc * alpha[:, None]
            vrow = (seq_start + offn)[:, None] * 3456 + 2304 + hoff + offd[None, :]
            v = tl.load(qkv_ptr + vrow, mask=nmask[:, None] & dmask[None, :], other=0.0)
            acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)

        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        tl.store(o_ptr + qrow, acc.to(tl.bfloat16), mask=mmask[:, None] & dmask[None, :])


@triton.jit
def _attn_fused(
    qkv_ptr, cos_ptr, sin_ptr, o_ptr, cu_ptr,
    S, SCALE,
    SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, SP)
    valid = offs < S
    cu = tl.load(cu_ptr + offs, mask=valid, other=0).to(tl.int32)
    cup = tl.load(cu_ptr + offs - 1, mask=valid & (offs > 0), other=0).to(tl.int32)
    lens = tl.where(valid, cu - cup, 0)
    nb = (lens + (BM - 1)) // BM
    cnb = tl.cumsum(nb, axis=0)
    sid = tl.sum((cnb <= pid).to(tl.int32), axis=0)
    if sid < S:
        sel = offs == sid
        start_blk = tl.sum(tl.where(sel, cnb - nb, 0), axis=0)
        seq_start = tl.sum(tl.where(sel, cup, 0), axis=0)
        L = tl.sum(tl.where(sel, lens, 0), axis=0)
        m0 = (pid - start_blk) * BM
        offm = m0 + tl.arange(0, BM)
        mmask = offm < L
        offd = tl.arange(0, BD)
        dmask = offd < 72
        lo = offd < 36
        offd2 = tl.where(lo, offd + 36, offd - 36)
        hoff = h * 72
        # ---- Q with rope ----
        rowq = (seq_start + offm)[:, None]
        cm = mmask[:, None] & dmask[None, :]
        base = rowq * 3456 + hoff
        cbase = rowq * 1152 + hoff
        c = tl.load(cos_ptr + cbase + offd[None, :], mask=cm, other=0.0).to(tl.float32)
        sn_ = tl.load(sin_ptr + cbase + offd[None, :], mask=cm, other=0.0).to(tl.float32)
        sn_ = tl.where(lo[None, :], -sn_, sn_)
        x1 = tl.load(qkv_ptr + base + offd[None, :], mask=cm, other=0.0).to(tl.float32)
        x2 = tl.load(qkv_ptr + base + offd2[None, :], mask=cm, other=0.0).to(tl.float32)
        q = ((x1 * c).to(tl.bfloat16).to(tl.float32) + (x2 * sn_).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

        m_i = tl.full([BM], -1.0e30, tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, BD], tl.float32)
        for s_ in range(0, tl.cdiv(L, BN)):
            offn = s_ * BN + tl.arange(0, BN)
            nmask = offn < L
            rown = (seq_start + offn)[:, None]
            cn = nmask[:, None] & dmask[None, :]
            kb = rown * 3456 + 1152 + hoff
            cb = rown * 1152 + hoff
            kc = tl.load(cos_ptr + cb + offd[None, :], mask=cn, other=0.0).to(tl.float32)
            ks = tl.load(sin_ptr + cb + offd[None, :], mask=cn, other=0.0).to(tl.float32)
            ks = tl.where(lo[None, :], -ks, ks)
            y1 = tl.load(qkv_ptr + kb + offd[None, :], mask=cn, other=0.0).to(tl.float32)
            y2 = tl.load(qkv_ptr + kb + offd2[None, :], mask=cn, other=0.0).to(tl.float32)
            k = ((y1 * kc).to(tl.bfloat16).to(tl.float32) + (y2 * ks).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            sc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            sc = sc.to(tl.bfloat16).to(tl.float32) * SCALE
            sc = sc.to(tl.bfloat16).to(tl.float32)
            sc = tl.where(nmask[None, :], sc, -1.0e30)
            mn = tl.maximum(m_i, tl.max(sc, 1))
            alpha = tl.exp(m_i - mn)
            p = tl.exp(sc - mn[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = mn
            acc = acc * alpha[:, None]
            v = tl.load(qkv_ptr + rown * 3456 + 2304 + hoff + offd[None, :], mask=cn, other=0.0)
            acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(o_ptr + rowq * 1152 + hoff + offd[None, :], (acc / l_safe[:, None]).to(tl.bfloat16), mask=cm)


@triton.jit
def _attn_split(
    q_ptr, k_ptr, qkv_ptr, o_ptr, cu_ptr,
    S, SCALE,
    SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0); h = tl.program_id(1)
    offs = tl.arange(0, SP)
    valid = offs < S
    cu = tl.load(cu_ptr + offs, mask=valid, other=0).to(tl.int32)
    cup = tl.load(cu_ptr + offs - 1, mask=valid & (offs > 0), other=0).to(tl.int32)
    lens = tl.where(valid, cu - cup, 0)
    nb = (lens + (BM - 1)) // BM
    cnb = tl.cumsum(nb, axis=0)
    sid = tl.sum((cnb <= pid).to(tl.int32), axis=0)
    if sid < S:
        sel = offs == sid
        start_blk = tl.sum(tl.where(sel, cnb - nb, 0), axis=0)
        seq_start = tl.sum(tl.where(sel, cup, 0), axis=0)
        L = tl.sum(tl.where(sel, lens, 0), axis=0)
        offm = (pid - start_blk) * BM + tl.arange(0, BM)
        mmask = offm < L
        dA = tl.arange(0, 64)          # dims 0..63
        dB = tl.arange(0, 16)          # dims 64..79 (72..79 masked)
        dBm = dB < 8
        hoff = h * 72
        rowq = (seq_start + offm)[:, None] * 1152 + hoff
        qA = tl.load(q_ptr + rowq + dA[None, :], mask=mmask[:, None], other=0.0)
        qB = tl.load(q_ptr + rowq + 64 + dB[None, :], mask=mmask[:, None] & dBm[None, :], other=0.0)
        m_i = tl.full([BM], -1.0e30, tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        accA = tl.zeros([BM, 64], tl.float32)
        accB = tl.zeros([BM, 16], tl.float32)
        for s_ in range(0, tl.cdiv(L, BN)):
            offn = s_ * BN + tl.arange(0, BN)
            nmask = offn < L
            rown = (seq_start + offn)[:, None] * 1152 + hoff
            kA = tl.load(k_ptr + rown + dA[None, :], mask=nmask[:, None], other=0.0)
            kB = tl.load(k_ptr + rown + 64 + dB[None, :], mask=nmask[:, None] & dBm[None, :], other=0.0)
            sc = tl.dot(qA, tl.trans(kA), out_dtype=tl.float32)
            sc = tl.dot(qB, tl.trans(kB), acc=sc, out_dtype=tl.float32)
            sc = sc.to(tl.bfloat16).to(tl.float32) * SCALE
            sc = sc.to(tl.bfloat16).to(tl.float32)
            sc = tl.where(nmask[None, :], sc, -1.0e30)
            mn = tl.maximum(m_i, tl.max(sc, 1))
            alpha = tl.exp(m_i - mn)
            p = tl.exp(sc - mn[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = mn
            pb = p.to(tl.bfloat16)
            vrow = (seq_start + offn)[:, None] * 3456 + 2304 + hoff
            vA = tl.load(qkv_ptr + vrow + dA[None, :], mask=nmask[:, None], other=0.0)
            vB = tl.load(qkv_ptr + vrow + 64 + dB[None, :], mask=nmask[:, None] & dBm[None, :], other=0.0)
            accA = accA * alpha[:, None] + tl.dot(pb, vA, out_dtype=tl.float32)
            accB = accB * alpha[:, None] + tl.dot(pb, vB, out_dtype=tl.float32)
        ls = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(o_ptr + rowq + dA[None, :], (accA / ls[:, None]).to(tl.bfloat16), mask=mmask[:, None])
        tl.store(o_ptr + rowq + 64 + dB[None, :], (accB / ls[:, None]).to(tl.bfloat16),
                 mask=mmask[:, None] & dBm[None, :])


_LAUNCH_CACHE = {}
_DIRECT_OK = [True]


def _cached(tag, sp):
    if not _DIRECT_OK[0]:
        return None
    return _LAUNCH_CACHE.get((tag, sp, torch.cuda.current_device()))


def _store(tag, sp, ck):
    """Keep the CompiledKernel Triton just returned so later calls skip JIT dispatch."""
    try:
        if ck is not None and ck.function is not None and ck.packed_metadata is not None:
            _LAUNCH_CACHE[(tag, sp, torch.cuda.current_device())] = ck
    except Exception:
        pass


def _fast(ck, grid, args):
    """Re-launch an already-compiled kernel, skipping Triton's JIT dispatch.

    Returns False (and permanently disables the fast path) if this Triton
    build's launcher signature differs from the one we were written against,
    so `run` can fall back to the ordinary JIT launch.
    """
    try:
        ck.run(grid[0], grid[1], 1, torch.cuda.current_stream().cuda_stream,
               ck.function, ck.packed_metadata, None, None, None, *args)
        return True
    except Exception:
        _DIRECT_OK[0] = False
        _LAUNCH_CACHE.clear()
        return False


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
) -> torch.Tensor:
    N = hidden_states.shape[0]
    S = cu_seqlens.shape[0]

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)

    if N == 0 or S == 0:
        return torch.zeros(N, EMBED, device=hidden_states.device, dtype=hidden_states.dtype)

    SP = max(16, triton.next_power_of_2(S))
    attn = torch.empty(N * EMBED, device=qkv.device, dtype=torch.bfloat16)
    scale = HD ** -0.5

    if N <= FUSE_N:
        grid = (triton.cdiv(N, 128) + S, NH)
        ck = _cached("fused", SP)
        if ck is None or not _fast(ck, grid,
                (qkv, cos, sin, attn, cu_seqlens, S, scale, SP, 128, 32, 128)):
            _store("fused", SP, _attn_fused[grid](
                qkv, cos, sin, attn, cu_seqlens, S, scale,
                SP=SP, BM=128, BN=32, BD=128, num_warps=4, num_stages=2,
            ))
        return F.linear(attn.view(N, EMBED), proj_weight, proj_bias)

    q = torch.empty(N * EMBED, device=qkv.device, dtype=torch.bfloat16)
    k = torch.empty(N * EMBED, device=qkv.device, dtype=torch.bfloat16)
    NTOT = N * EMBED
    BLOCK = 256
    rg = triton.cdiv(NTOT, BLOCK)
    ck = _cached("rope", 0)
    if ck is None or not _fast(ck, (rg, 1), (qkv, cos, sin, q, k, NTOT, BLOCK)):
        _store("rope", 0, _rope_kernel[(rg,)](
            qkv, cos, sin, q, k, NTOT, BLOCK=BLOCK, num_warps=2))

    if N // S <= 80:
        BM, BN = 64, 32
    else:
        BM, BN = 128, 64
    grid = (triton.cdiv(N, BM) + S, NH)
    ck = _cached(("attn", BM, BN), SP)
    if ck is None or not _fast(ck, grid,
            (q, k, qkv, attn, cu_seqlens, S, scale, SP, BM, BN)):
        _store(("attn", BM, BN), SP, _attn_split[grid](
            q, k, qkv, attn, cu_seqlens, S, scale,
            SP=SP, BM=BM, BN=BN, num_warps=4, num_stages=2,
        ))

    return F.linear(attn.view(N, EMBED), proj_weight, proj_bias)
