import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# RWKV WKV forward with exponential (max-state) stabilization.
#
# The recurrence is elementwise over (batch, channel); the only sequential
# dependence is along `t`.  Channels are independent, so a program owns BLOCK
# channels of one batch and marches over a contiguous slice of t.
#
# To expose more parallelism than B*ceil(H/BLOCK) lanes, the sequence is split
# into NC chunks.  Chunk states compose exactly under the log-sum-exp semiring:
#
#   state = (m, n, d)   representing   N = n*exp(m),  D = d*exp(m)
#   advancing a state by L steps of decay w:   m <- m + L*w
#   combine(prefix, local):
#       m' = max(prefix.m + L*w, local.m)
#       n' = prefix.n*exp(prefix.m + L*w - m') + local.n*exp(local.m - m')
#
# so phase A computes per-chunk local states, phase B scans them (tiny), and
# phase C replays each chunk from its exact prefix state and writes outputs.
#
# Numerics: the reference runs in float32 with unfused mul/add (torch emits
# separate v_mul/v_add for `e1*n + e2*v`) and ROCm's expf.  Both are reproduced
# bit-for-bit here: `enable_fp_fusion=False` on every launch stops Triton from
# contracting mul+add into FMA, and _expf below is the exact gfx950 expf
# sequence (Cody-Waite reduction in log2 space + hardware v_exp_f32 + ldexp).
# ---------------------------------------------------------------------------


# gfx950 ::expf constants, read out of the compiled ISA.
_LOG2E_HI = 1.4426950216293335      # 0x3fb8aa3b
_LOG2E_LO = 1.925962855864327e-08   # 0x32a5705f
_EXP_LO = -103.2789306640625        # 0xc2ce8ed0  (underflow-to-zero cutoff)


@triton.jit
def _expf(x):
    """Bit-exact reproduction of ROCm ::expf(float) on gfx950."""
    t = x * _LOG2E_HI
    e = libdevice.fma(x, _LOG2E_HI, -t)
    n = libdevice.rint(t)
    e = libdevice.fma(x, _LOG2E_LO, e)
    t = t - n
    t = t + e
    r = tl.exp2(t)
    r = libdevice.ldexp(r, n.to(tl.int32))
    # The reference never evaluates exp(x) for x > 0 (arguments are always
    # `something - max(..., something)`), so only the underflow leg matters.
    return tl.where(x >= _EXP_LO, r, 0.0)


@triton.jit
def _scan_kernel(
    K, V, TD, TF,
    PM, PN, PD,          # prefix state  [NC, B, H]
    OUT,
    OM, ON, OD,          # written state [NC, B, H]
    B, T, H, L,
    nblk,
    HAS_PREFIX: tl.constexpr,
    WRITE_OUT: tl.constexpr,
    WRITE_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_h // nblk
    hb = pid_h % nblk

    offs = hb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H

    td = tl.load(TD + offs, mask=mask, other=0.0)
    w = -_expf(td)
    tf = tl.load(TF + offs, mask=mask, other=0.0)

    base_s = (pid_c * B + b) * H + offs

    if HAS_PREFIX:
        m = tl.load(PM + base_s, mask=mask, other=0.0)
        n = tl.load(PN + base_s, mask=mask, other=0.0)
        d = tl.load(PD + base_s, mask=mask, other=0.0)
    else:
        m = tl.full((BLOCK,), -1e38, tl.float32)
        n = tl.zeros((BLOCK,), tl.float32)
        d = tl.zeros((BLOCK,), tl.float32)

    t_start = pid_c * L
    t_end = tl.minimum(t_start + L, T)

    p = (b * T + t_start) * H + offs
    for _t in range(t_start, t_end):
        k = tl.load(K + p, mask=mask, other=0.0)
        v = tl.load(V + p, mask=mask, other=0.0)

        if WRITE_OUT:
            # max(m, k+tf): exactly one side of the pair exponentiates to
            # exp(0) == 1.0, so a select on the comparison replaces one exp
            # and is bit-identical to evaluating it.
            kt = k + tf
            ge = m >= kt
            diff = tl.where(ge, kt - m, m - kt)
            ex = _expf(diff)
            e1 = tl.where(ge, 1.0, ex)
            e2 = tl.where(ge, ex, 1.0)
            num = e1 * n + e2 * v
            den = e1 * d + e2
            tl.store(OUT + p, num / den, mask=mask)

        m2 = m + w
        ge2 = m2 >= k
        diff2 = tl.where(ge2, k - m2, m2 - k)
        ex2 = _expf(diff2)
        f1 = tl.where(ge2, 1.0, ex2)
        f2 = tl.where(ge2, ex2, 1.0)
        n = f1 * n + f2 * v
        d = f1 * d + f2
        m = tl.maximum(m2, k)

        p += H

    if WRITE_STATE:
        tl.store(OM + base_s, m, mask=mask)
        tl.store(ON + base_s, n, mask=mask)
        tl.store(OD + base_s, d, mask=mask)


@triton.jit
def _combine_kernel(
    TD, MS, NS, DS,
    LM, LN, LD,          # per-chunk local states   [NC, B, H]
    PM, PN, PD,          # exclusive prefix states  [NC, B, H]
    OM, ON, OD,          # final states             [B, H]
    B, T, H, L, NC,
    nblk,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // nblk
    hb = pid % nblk

    offs = hb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H

    td = tl.load(TD + offs, mask=mask, other=0.0)
    w = -_expf(td)

    m = tl.load(MS + b * H + offs, mask=mask, other=0.0)
    n = tl.load(NS + b * H + offs, mask=mask, other=0.0)
    d = tl.load(DS + b * H + offs, mask=mask, other=0.0)

    for c in range(NC):
        idx = (c * B + b) * H + offs
        tl.store(PM + idx, m, mask=mask)
        tl.store(PN + idx, n, mask=mask)
        tl.store(PD + idx, d, mask=mask)

        lm = tl.load(LM + idx, mask=mask, other=0.0)
        ln = tl.load(LN + idx, mask=mask, other=0.0)
        ld = tl.load(LD + idx, mask=mask, other=0.0)

        # advancing the prefix state across a whole chunk is L applications of
        # `m += w`; done as a single multiply here (see _combine note below).
        Lc = tl.minimum(L, T - c * L)
        ma = m + Lc.to(tl.float32) * w
        mn = tl.maximum(ma, lm)
        e1 = _expf(ma - mn)
        e2 = _expf(lm - mn)
        n = e1 * n + e2 * ln
        d = e1 * d + e2 * ld
        m = mn

    tl.store(OM + b * H + offs, m, mask=mask)
    tl.store(ON + b * H + offs, n, mask=mask)
    tl.store(OD + b * H + offs, d, mask=mask)


_BLOCK = 64
_NUM_WARPS = 1
_NUM_STAGES = 2
_TARGET_PROGRAMS = 8192
_MIN_CHUNK = 128


def _plan(B, T, H):
    nblk = triton.cdiv(H, _BLOCK)
    per_chunk = B * nblk
    nc = max(1, -(-_TARGET_PROGRAMS // per_chunk))
    nc = min(nc, max(1, T // _MIN_CHUNK))
    L = -(-T // nc)
    nc = -(-T // L)
    return nblk, nc, L


@torch.no_grad()
def run(
    time_decay: torch.Tensor,
    key: torch.Tensor,
    time_first: torch.Tensor,
    value: torch.Tensor,
    max_state: torch.Tensor,
    num_state: torch.Tensor,
    den_state: torch.Tensor,
):
    B, T, H = key.shape
    dev = key.device

    key = key.contiguous()
    value = value.contiguous()
    time_decay = time_decay.contiguous()
    time_first = time_first.contiguous()
    max_state = max_state.contiguous()
    num_state = num_state.contiguous()
    den_state = den_state.contiguous()

    output = torch.empty((B, T, H), dtype=torch.float32, device=dev)

    if T == 0:
        return output, max_state.clone(), num_state.clone(), den_state.clone()

    m_out = torch.empty((B, H), dtype=torch.float32, device=dev)
    n_out = torch.empty((B, H), dtype=torch.float32, device=dev)
    d_out = torch.empty((B, H), dtype=torch.float32, device=dev)

    nblk, nc, L = _plan(B, T, H)

    if nc == 1:
        _scan_kernel[(1, B * nblk)](
            key, value, time_decay, time_first,
            max_state, num_state, den_state,
            output,
            m_out, n_out, d_out,
            B, T, H, L, nblk,
            HAS_PREFIX=True, WRITE_OUT=True, WRITE_STATE=True,
            BLOCK=_BLOCK, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
            enable_fp_fusion=False,
        )
        return output, m_out, n_out, d_out

    lm = torch.empty((nc, B, H), dtype=torch.float32, device=dev)
    ln = torch.empty((nc, B, H), dtype=torch.float32, device=dev)
    ld = torch.empty((nc, B, H), dtype=torch.float32, device=dev)
    pm = torch.empty((nc, B, H), dtype=torch.float32, device=dev)
    pn = torch.empty((nc, B, H), dtype=torch.float32, device=dev)
    pd = torch.empty((nc, B, H), dtype=torch.float32, device=dev)

    grid = (nc, B * nblk)

    # phase A: per-chunk local states, each starting from the identity state
    _scan_kernel[grid](
        key, value, time_decay, time_first,
        lm, ln, ld,          # unused (HAS_PREFIX=False)
        output,
        lm, ln, ld,
        B, T, H, L, nblk,
        HAS_PREFIX=False, WRITE_OUT=False, WRITE_STATE=True,
        BLOCK=_BLOCK, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        enable_fp_fusion=False,
    )

    # phase B: sequential scan over chunk states (NC steps, tiny)
    _combine_kernel[(B * nblk,)](
        time_decay, max_state, num_state, den_state,
        lm, ln, ld,
        pm, pn, pd,
        m_out, n_out, d_out,
        B, T, H, L, nc, nblk,
        BLOCK=_BLOCK, num_warps=_NUM_WARPS, enable_fp_fusion=False,
    )

    # phase C: replay each chunk from its exact prefix state, writing outputs
    _scan_kernel[grid](
        key, value, time_decay, time_first,
        pm, pn, pd,
        output,
        lm, ln, ld,          # unused (WRITE_STATE=False)
        B, T, H, L, nblk,
        HAS_PREFIX=True, WRITE_OUT=True, WRITE_STATE=False,
        BLOCK=_BLOCK, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        enable_fp_fusion=False,
    )

    return output, m_out, n_out, d_out
