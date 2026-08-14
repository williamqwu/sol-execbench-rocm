import torch
import triton
import triton.language as tl


@triton.jit
def _bf(x):
    # round a float32 value through bfloat16 (RTNE), matching torch's
    # elementwise bf16 arithmetic
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _prep_kernel(
    K, V, G, BETA, VT, KCD,
    T, NCT,
    HK: tl.constexpr, HV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    b = i_bh // HV
    hv = i_bh % HV
    kh = hv % HK

    idx = tl.arange(0, BT)
    ot = i_c * BT + idx
    m = ot < T
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)

    # ---- key, l2-normalized (emulating torch bf16 rounding) ----
    k = tl.load(K + ((b * HK + kh) * T + ot[:, None]) * DK + dk[None, :],
                mask=m[:, None], other=0.0).to(tl.float32)
    sq = _bf(k * k)
    s = tl.sum(sq, 1)
    s = _bf(s)
    s = _bf(s + 1e-6)
    inv = _bf(tl.rsqrt(s))
    k = _bf(k * inv[:, None])

    g = tl.load(G + (b * HV + hv) * T + ot, mask=m, other=0.0).to(tl.float32)
    beta = tl.load(BETA + (b * HV + hv) * T + ot, mask=m, other=0.0).to(tl.float32)

    gc = tl.cumsum(g, 0)

    kb = k * beta[:, None]
    dm = tl.exp(gc[:, None] - gc[None, :])

    # A = strictly-lower( -(k_beta @ k^T) * decay )
    Bm = -tl.dot(kb, tl.trans(k), input_precision="ieee") * dm
    Bm = tl.where(idx[:, None] > idx[None, :], Bm, 0.0)

    # (I - A)^-1 = prod_j (I + A^(2^j)); A is nilpotent of index BT
    S = tl.where(idx[:, None] == idx[None, :], 1.0, 0.0) + Bm
    P = Bm
    for _ in tl.static_range(5):
        P = tl.dot(P, P, input_precision="ieee")
        S = S + tl.dot(P, S, input_precision="ieee")

    v = tl.load(V + ((b * HV + hv) * T + ot[:, None]) * DV + dv[None, :],
                mask=m[:, None], other=0.0).to(tl.float32)
    vb = v * beta[:, None]
    kg = kb * tl.exp(g)[:, None]

    vt = tl.dot(S, vb, input_precision="ieee")
    kcd = tl.dot(S, kg, input_precision="ieee")

    tl.store(VT + ((b * HV + hv) * NCT + ot[:, None]) * DV + dv[None, :], vt)
    tl.store(KCD + ((b * HV + hv) * NCT + ot[:, None]) * DK + dk[None, :], kcd)


@triton.jit
def _scan_kernel(
    Q, K, G, VT, KCD, O,
    T, NCT, NC, scale,
    HK: tl.constexpr, HV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    BV: tl.constexpr, BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    b = i_bh // HV
    hv = i_bh % HV
    kh = hv % HK

    idx = tl.arange(0, BT)
    dk = tl.arange(0, DK)
    dvb = i_v * BV + tl.arange(0, BV)

    state = tl.zeros([DK, BV], dtype=tl.float32)

    for ic in range(NC):
        ot = ic * BT + idx
        m = ot < T

        q = tl.load(Q + ((b * HK + kh) * T + ot[:, None]) * DK + dk[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        sq = _bf(q * q)
        s = _bf(tl.sum(sq, 1))
        s = _bf(s + 1e-6)
        q = _bf(q * _bf(tl.rsqrt(s))[:, None]) * scale

        k = tl.load(K + ((b * HK + kh) * T + ot[:, None]) * DK + dk[None, :],
                    mask=m[:, None], other=0.0).to(tl.float32)
        sq = _bf(k * k)
        s = _bf(tl.sum(sq, 1))
        s = _bf(s + 1e-6)
        k = _bf(k * _bf(tl.rsqrt(s))[:, None])

        g = tl.load(G + (b * HV + hv) * T + ot, mask=m, other=0.0).to(tl.float32)
        gc = tl.cumsum(g, 0)

        dm = tl.exp(gc[:, None] - gc[None, :])
        ai = tl.dot(q, tl.trans(k), input_precision="ieee") * dm
        ai = tl.where(idx[:, None] >= idx[None, :], ai, 0.0)

        vt = tl.load(VT + ((b * HV + hv) * NCT + ot[:, None]) * DV + dvb[None, :])
        kcd = tl.load(KCD + ((b * HV + hv) * NCT + ot[:, None]) * DK + dk[None, :])

        vn = vt - tl.dot(kcd, state, input_precision="ieee")

        eg = tl.exp(g)
        o = tl.dot(q * eg[:, None], state, input_precision="ieee")
        o += tl.dot(ai, vn, input_precision="ieee")

        tl.store(O + ((b * T + ot[:, None]) * HV + hv) * DV + dvb[None, :],
                 o.to(O.dtype.element_ty), mask=m[:, None])

        glast = tl.sum(tl.where(idx == BT - 1, g, 0.0))
        kd = k * tl.exp(glast - g)[:, None]
        state = state * tl.exp(glast) + tl.dot(tl.trans(kd), vn, input_precision="ieee")


@torch.no_grad()
def run(query, key, value, g, beta, scale):
    B, HK, T, DK = key.shape
    HV = value.shape[1]
    DV = value.shape[-1]
    BT = 64
    NC = (T + BT - 1) // BT
    NCT = NC * BT

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    dev = query.device
    vt = torch.empty((B, HV, NCT, DV), dtype=torch.float32, device=dev)
    kcd = torch.empty((B, HV, NCT, DK), dtype=torch.float32, device=dev)

    _prep_kernel[(NC, B * HV)](
        key, value, g, beta, vt, kcd,
        T, NCT,
        HK=HK, HV=HV, DK=DK, DV=DV, BT=BT,
        num_warps=4, num_stages=1,
    )

    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=dev)

    BV = 64
    if B * HV * (DV // BV) < 128:
        BV = 32

    _scan_kernel[(B * HV, DV // BV)](
        query, key, g, vt, kcd, out,
        T, NCT, NC, scale,
        HK=HK, HV=HV, DK=DK, DV=DV, BV=BV, BT=BT,
        num_warps=4, num_stages=1,
    )
    return out
