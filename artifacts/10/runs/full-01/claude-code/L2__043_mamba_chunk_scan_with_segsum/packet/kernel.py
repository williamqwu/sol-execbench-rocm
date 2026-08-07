import torch
import triton
import triton.language as tl

# Constants fixed by the problem definition
H = 16       # num_heads
P = 64       # head_dim
N = 256      # state_size
CS = 256     # chunk_size


@triton.jit
def _split(x):
    """Split an fp32 tile into two bf16 tiles (hi + lo) with hi + lo ~= x to
    ~16 mantissa bits, so bf16 MFMA reaches ~fp32 accuracy.  Non-finite lanes
    keep hi = +-inf / nan and lo = 0 so they propagate like the fp32 reference
    instead of turning into nan via inf - inf."""
    hi = x.to(tl.bfloat16)
    hif = hi.to(tl.float32)
    lo = tl.where(hif == hif, x - hif, 0.0).to(tl.bfloat16)
    return hi, lo


@triton.jit
def _safe_scale(x):
    """Largest-magnitude element of the tile, as a scale that never overflows
    when x is divided by it.  Falls back to 1 for zero / non-finite tiles."""
    mx = tl.max(tl.abs(x))
    ok = (mx > 0.0) & (mx < 3.0e38)
    return tl.where(ok, mx, 1.0)


@triton.jit
def _k_chunk_state(
    A_ptr, HS_ptr, B_ptr, ACS_ptr, ST_ptr,
    L, NC,
    BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    SBLK: tl.constexpr, H_: tl.constexpr,
):
    pid = tl.program_id(0)          # b * NC + c
    h = tl.program_id(1)
    c = pid % NC
    b = pid // NC

    offs_t = tl.arange(0, BLOCK_T)
    tg = c * BLOCK_T + offs_t
    mt = tg < L

    a = tl.load(A_ptr + (b * H_ + h) * L + tg, mask=mt, other=0.0).to(tl.float32)
    acs = tl.cumsum(a, 0)
    tl.store(ACS_ptr + ((b * H_ + h) * NC + c) * BLOCK_T + offs_t, acs)

    e = tl.sum(tl.where(offs_t == BLOCK_T - 1, acs, 0.0), 0)
    decay = tl.exp(e - acs)  # [T]

    offs_d = tl.arange(0, BLOCK_D)

    hs = tl.load(
        HS_ptr + ((b * L + tg[:, None]) * H_ + h) * BLOCK_D + offs_d[None, :],
        mask=mt[:, None], other=0.0,
    ).to(tl.float32)
    hsd = hs * decay[:, None]
    sc = _safe_scale(hsd)
    hi, lo = _split(hsd / sc)

    for s0 in tl.range(0, BLOCK_S, SBLK):
        offs_s = s0 + tl.arange(0, SBLK)
        # B transposed tile: [SBLK, T]
        bt = tl.load(
            B_ptr + (b * L + tg[None, :]) * BLOCK_S + offs_s[:, None],
            mask=mt[None, :], other=0.0,
        )
        st = tl.dot(bt, hi)
        st = tl.dot(bt, lo, st)
        tl.store(
            ST_ptr + (((b * NC + c) * H_ + h) * BLOCK_S + offs_s[:, None]) * BLOCK_D
            + offs_d[None, :],
            st * sc,
        )


@triton.jit
def _k_scan(
    ACS_ptr, ST_ptr, OUTST_ptr, INIT_ptr, FIN_ptr,
    NC,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, P_: tl.constexpr,
    H_: tl.constexpr, T_: tl.constexpr,
):
    pid = tl.program_id(0)   # b * H + h
    dblk = tl.program_id(1)
    h = pid % H_
    b = pid // H_

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)

    # Reproduce the reference's *pairwise* inter-chunk recurrence exactly:
    #   new_states[i] = sum_{j<=i} exp(cs[i] - cs[j]) * sw[j]
    # with sw[0] = initial_states, sw[j] = states[j-1], and cs the cumulative
    # sum of the per-chunk A totals (cs[0] = 0).  A plain sequential scan is
    # algebraically identical but overflows to inf on chunks whose decay spikes
    # and then never recovers; the pairwise form does not.  num_chunks is small
    # (<= 16 for every workload here) so the O(nc^2) cost is negligible.
    sw0 = tl.load(
        INIT_ptr + ((b * H_ + h) * P_ + offs_d[None, :]) * BLOCK_S + offs_s[:, None]
    ).to(tl.float32)

    csi = 0.0
    for i in range(0, NC + 1):
        acc = tl.exp(csi) * sw0  # j = 0 term (cs[0] = 0)
        csj = 0.0
        for j in range(1, NC + 1):
            csj += tl.load(ACS_ptr + ((b * H_ + h) * NC + (j - 1)) * T_ + (T_ - 1))
            if j <= i:
                sj = tl.load(
                    ST_ptr + (((b * NC + (j - 1)) * H_ + h) * BLOCK_S + offs_s[:, None]) * P_
                    + offs_d[None, :]
                )
                acc += tl.exp(csi - csj) * sj

        if i < NC:
            tl.store(
                OUTST_ptr + (((b * NC + i) * H_ + h) * BLOCK_S + offs_s[:, None]) * P_
                + offs_d[None, :],
                acc,
            )
            csi += tl.load(ACS_ptr + ((b * H_ + h) * NC + i) * T_ + (T_ - 1))
        else:
            tl.store(
                FIN_ptr + ((b * H_ + h) * P_ + offs_d[None, :]) * BLOCK_S + offs_s[:, None],
                acc.to(FIN_ptr.dtype.element_ty),
            )


@triton.jit
def _k_out(
    HS_ptr, B_ptr, C_ptr, D_ptr, ACS_ptr, ST_ptr, OUT_ptr,
    L, NC,
    BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr, SBLK: tl.constexpr, H_: tl.constexpr,
):
    pid = tl.program_id(0)          # b * NC + c
    h = tl.program_id(1)
    m0 = tl.program_id(2) * BLOCK_M
    c = pid % NC
    b = pid // NC

    offs_i = m0 + tl.arange(0, BLOCK_M)
    offs_j = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    ti = c * BLOCK_T + offs_i
    tj = c * BLOCK_T + offs_j
    mi = ti < L
    mj = tj < L

    abase = ACS_ptr + ((b * H_ + h) * NC + c) * BLOCK_T
    acs_i = tl.load(abase + offs_i)
    acs_j = tl.load(abase + offs_j)

    g = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
    yo = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for s0 in tl.range(0, BLOCK_S, SBLK):
        offs_s = s0 + tl.arange(0, SBLK)
        # C tile [M, SBLK]
        ct = tl.load(
            C_ptr + (b * L + ti[:, None]) * BLOCK_S + offs_s[None, :],
            mask=mi[:, None], other=0.0,
        )
        # B tile transposed [SBLK, J]
        bt = tl.load(
            B_ptr + (b * L + tj[None, :]) * BLOCK_S + offs_s[:, None],
            mask=mj[None, :], other=0.0,
        )
        g = tl.dot(ct, bt, g)

        st = tl.load(
            ST_ptr + (((b * NC + c) * H_ + h) * BLOCK_S + offs_s[:, None]) * BLOCK_D
            + offs_d[None, :]
        )
        ssc = _safe_scale(st)
        sthi, stlo = _split(st / ssc)
        t = tl.dot(ct, sthi)
        t = tl.dot(ct, stlo, t)
        yo += t * ssc

    lmat = tl.where(offs_i[:, None] >= offs_j[None, :],
                    tl.exp(acs_i[:, None] - acs_j[None, :]), 0.0)
    m = g * lmat
    msc = _safe_scale(m)
    mhi, mlo = _split(m / msc)

    hsj = tl.load(
        HS_ptr + ((b * L + tj[:, None]) * H_ + h) * BLOCK_D + offs_d[None, :],
        mask=mj[:, None], other=0.0,
    )
    y = tl.dot(mhi, hsj)
    y = tl.dot(mlo, hsj, y)
    y = y * msc

    y += yo * tl.exp(acs_i)[:, None]

    hsi = tl.load(
        HS_ptr + ((b * L + ti[:, None]) * H_ + h) * BLOCK_D + offs_d[None, :],
        mask=mi[:, None], other=0.0,
    ).to(tl.float32)
    d = tl.load(D_ptr + h).to(tl.float32)
    y += d * hsi

    tl.store(
        OUT_ptr + (b * L + ti[:, None]) * (H_ * BLOCK_D) + h * BLOCK_D + offs_d[None, :],
        y.to(OUT_ptr.dtype.element_ty),
        mask=mi[:, None],
    )


@torch.no_grad()
def run(hidden_states, A, B, C, D, initial_states):
    b, L, nh, hd = hidden_states.shape
    nc = (L + CS - 1) // CS

    hidden_states = hidden_states.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    D = D.contiguous()
    initial_states = initial_states.contiguous()

    dev = hidden_states.device
    acs = torch.empty((b, nh, nc, CS), device=dev, dtype=torch.float32)
    states = torch.empty((b, nc, nh, N, hd), device=dev, dtype=torch.float32)
    states_out = torch.empty_like(states)
    final = torch.empty((b, nh, hd, N), device=dev, dtype=torch.bfloat16)
    out = torch.empty((b, L, nh * hd), device=dev, dtype=torch.bfloat16)

    _k_chunk_state[(b * nc, nh)](
        A, hidden_states, B, acs, states,
        L, nc,
        BLOCK_T=CS, BLOCK_S=N, BLOCK_D=hd, SBLK=64, H_=nh,
        num_warps=4, num_stages=1,
    )

    BD = 16
    _k_scan[(b * nh, hd // BD)](
        acs, states, states_out, initial_states, final,
        nc,
        BLOCK_S=N, BLOCK_D=BD, P_=hd, H_=nh, T_=CS,
        num_warps=4, num_stages=1,
    )

    BM = 64
    _k_out[(b * nc, nh, CS // BM)](
        hidden_states, B, C, D, acs, states_out, out,
        L, nc,
        BLOCK_M=BM, BLOCK_T=CS, BLOCK_S=N, BLOCK_D=hd, SBLK=64, H_=nh,
        num_warps=4, num_stages=1,
    )

    return out, final
