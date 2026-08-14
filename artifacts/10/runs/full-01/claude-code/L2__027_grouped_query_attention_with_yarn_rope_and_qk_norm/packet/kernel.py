import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused RMSNorm (over full hidden dim) + YARN RoPE for Q and K, one program
# per token.  Reproduces the reference's bf16 intermediate rounding exactly.
# ---------------------------------------------------------------------------


@triton.jit
def _qk_norm_rope(
    QIN, KIN, QOUT, KOUT, POS, INVF, QW, KW,
    S, ps0, ps1,
    eps, af,
    NHQ: tl.constexpr, BHQ: tl.constexpr, NHK: tl.constexpr,
    HD: tl.constexpr,          # half head dim = 64
    QHID: tl.constexpr, KHID: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid - b * S

    d = tl.arange(0, HD)
    pos = tl.load(POS + b * ps0 + s * ps1).to(tl.float32)
    inv = tl.load(INVF + d)
    freq = inv * pos
    cos = (tl.cos(freq) * af).to(tl.bfloat16)
    sin = (tl.sin(freq) * af).to(tl.bfloat16)
    cosf = cos.to(tl.float32)[None, :]
    sinf = sin.to(tl.float32)[None, :]

    # ------------------------------------------------------------------ Q
    h = tl.arange(0, BHQ)[:, None]
    dd = d[None, :]
    hm = h < NHQ
    o1 = h * (2 * HD) + dd
    o2 = o1 + HD

    base = QIN + pid.to(tl.int64) * QHID
    a = tl.load(base + o1, mask=hm, other=0.0).to(tl.float32)
    c = tl.load(base + o2, mask=hm, other=0.0).to(tl.float32)
    ss = tl.sum(a * a) + tl.sum(c * c)
    rstd = tl.math.rsqrt(ss / QHID + eps)
    w1 = tl.load(QW + o1, mask=hm, other=0.0).to(tl.float32)
    w2 = tl.load(QW + o2, mask=hm, other=0.0).to(tl.float32)
    a = ((a * rstd) * w1).to(tl.bfloat16).to(tl.float32)
    c = ((c * rstd) * w2).to(tl.bfloat16).to(tl.float32)
    # rope with per-op bf16 rounding
    r1 = ((a * cosf).to(tl.bfloat16).to(tl.float32)
          + ((-c) * sinf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    r2 = ((c * cosf).to(tl.bfloat16).to(tl.float32)
          + (a * sinf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    ob = QOUT + pid.to(tl.int64) * QHID
    tl.store(ob + o1, r1, mask=hm)
    tl.store(ob + o2, r2, mask=hm)

    # ------------------------------------------------------------------ K
    hk = tl.arange(0, NHK)[:, None]
    k1 = hk * (2 * HD) + dd
    k2 = k1 + HD
    kbase = KIN + pid.to(tl.int64) * KHID
    a = tl.load(kbase + k1).to(tl.float32)
    c = tl.load(kbase + k2).to(tl.float32)
    ss = tl.sum(a * a) + tl.sum(c * c)
    rstd = tl.math.rsqrt(ss / KHID + eps)
    w1 = tl.load(KW + k1).to(tl.float32)
    w2 = tl.load(KW + k2).to(tl.float32)
    a = ((a * rstd) * w1).to(tl.bfloat16).to(tl.float32)
    c = ((c * rstd) * w2).to(tl.bfloat16).to(tl.float32)
    r1 = ((a * cosf).to(tl.bfloat16).to(tl.float32)
          + ((-c) * sinf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    r2 = ((c * cosf).to(tl.bfloat16).to(tl.float32)
          + (a * sinf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    kob = KOUT + pid.to(tl.int64) * KHID
    tl.store(kob + k1, r1)
    tl.store(kob + k2, r2)


# ---------------------------------------------------------------------------
# Pass 1: derive, per (batch, query-block), the first and last key column that
# is not fully masked. This reads the mask tensor and makes no assumption about
# its structure -- an arbitrary additive mask gives correct (possibly full)
# bounds. Every key inside [lo, hi) is still scored normally; only blocks where
# the mask is -inf for every row are skipped, which cannot change the result.
# ---------------------------------------------------------------------------


@triton.jit
def _mask_bounds(M, LO, HI, S, smb, smm, smn,
                 BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    rowm = offs_m[:, None] < S
    base = M + b.to(tl.int64) * smb + offs_m[:, None] * smm
    lo = S
    hi = 0
    for start_n in tl.range(0, S, BN):
        cn = start_n + offs_n
        mm = tl.load(base + cn[None, :] * smn,
                     mask=rowm & (cn[None, :] < S), other=float("-inf"))
        any_col = tl.max((mm > float("-inf")).to(tl.int32), 0) > 0
        lo = tl.minimum(lo, tl.min(tl.where(any_col, cn, S)))
        hi = tl.maximum(hi, tl.max(tl.where(any_col, cn + 1, 0)))
    lo = tl.minimum(lo, hi)
    nb = tl.num_programs(0)
    tl.store(LO + b * nb + pid_m, (lo // BN) * BN)
    tl.store(HI + b * nb + pid_m, hi)


# ---------------------------------------------------------------------------
# Flash attention with GQA, external additive mask.
# ---------------------------------------------------------------------------


@triton.jit
def _attn_fwd(
    Q, K, V, M, O, LO, HI,
    S, scaling,
    smb, smm, smn,
    NH: tl.constexpr, GQ: tl.constexpr, D: tl.constexpr,
    QHID: tl.constexpr, KHID: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NH
    h = pid_bh - b * NH
    kvh = h // GQ

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BN)

    tok0 = b.to(tl.int64) * S
    qp = Q + (tok0 + offs_m[:, None]).to(tl.int64) * QHID + h * D + offs_d[None, :]
    q = tl.load(qp, mask=offs_m[:, None] < S, other=0.0)

    kbase = K + tok0 * KHID + kvh * D
    vbase = V + tok0 * KHID + kvh * D
    mbase = M + b.to(tl.int64) * smb + (pid_m * BM + tl.arange(0, BM))[:, None] * smm

    m_i = tl.full([BM], -1e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634

    rowm = offs_m[:, None] < S
    nb = tl.num_programs(0)
    lo = tl.load(LO + b * nb + pid_m)
    hi = tl.load(HI + b * nb + pid_m)
    # Branch-free inner loop over only the key blocks that pass 1 found live.
    for start_n in tl.range(lo, hi, BN):
        cn = start_n + offs_n
        kmask = cn[:, None] < S
        mm = tl.load(mbase + cn[None, :] * smn,
                     mask=rowm & (cn[None, :] < S),
                     other=float("-inf")).to(tl.float32)
        k = tl.load(kbase + cn[:, None].to(tl.int64) * KHID + offs_d[None, :],
                    mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k))
        # emulate: bf16 matmul output -> * scaling (bf16) -> + mask (bf16)
        s = (s.to(tl.bfloat16).to(tl.float32) * scaling).to(tl.bfloat16).to(tl.float32)
        s = (s + mm).to(tl.bfloat16).to(tl.float32)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(vbase + cn[:, None].to(tl.int64) * KHID + offs_d[None, :],
                    mask=kmask, other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    op = O + (tok0 + offs_m[:, None]).to(tl.int64) * QHID + h * D + offs_d[None, :]
    tl.store(op, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)


import os

_ATTN_CFG_ENV = os.environ.get("ATTN_CFG")


def _attn_cfg(B, S):
    if _ATTN_CFG_ENV:
        return tuple(int(x) for x in _ATTN_CFG_ENV.split(","))
    # Long sequences favour a narrower key tile (more skipped mask blocks,
    # better occupancy); short ones favour the wider tile.
    if S >= 1536:
        return (128, 32, 4, 2)
    return (128, 64, 4, 3)


HIDDEN = 5120
NHQ = 40
NHK = 8
HEAD_DIM = 128
KHID = NHK * HEAD_DIM


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
    attention_factor: float,
    scaling: float,
):
    B, S, _ = hidden_states.shape
    M = B * S
    hs = hidden_states.reshape(M, HIDDEN)

    q_lin = torch.nn.functional.linear(hs, q_proj_weight)
    k_lin = torch.nn.functional.linear(hs, k_proj_weight)
    v_lin = torch.nn.functional.linear(hs, v_proj_weight)

    qr = torch.empty_like(q_lin)
    kr = torch.empty_like(k_lin)

    _qk_norm_rope[(M,)](
        q_lin, k_lin, qr, kr, position_ids, inv_freq, q_norm_weight, k_norm_weight,
        S, position_ids.stride(0), position_ids.stride(1),
        rms_norm_eps, attention_factor,
        NHQ=NHQ, BHQ=64, NHK=NHK, HD=HEAD_DIM // 2,
        QHID=HIDDEN, KHID=KHID,
        num_warps=1, num_stages=2,
    )

    attn = torch.empty_like(q_lin)
    BM, BN, nw, ns = _attn_cfg(B, S)
    nblk = triton.cdiv(S, BM)
    lo = torch.empty(B * nblk, dtype=torch.int32, device=q_lin.device)
    hi = torch.empty(B * nblk, dtype=torch.int32, device=q_lin.device)
    _mask_bounds[(nblk, B)](
        attention_mask, lo, hi, S,
        attention_mask.stride(0), attention_mask.stride(2), attention_mask.stride(3),
        BM=BM, BN=BN, num_warps=4,
    )
    grid = (nblk, B * NHQ)
    _attn_fwd[grid](
        qr, kr, v_lin, attention_mask, attn, lo, hi,
        S, scaling,
        attention_mask.stride(0), attention_mask.stride(2), attention_mask.stride(3),
        NH=NHQ, GQ=NHQ // NHK, D=HEAD_DIM,
        QHID=HIDDEN, KHID=KHID,
        BM=BM, BN=BN,
        num_warps=nw, num_stages=ns,
    )

    out = torch.nn.functional.linear(attn, o_proj_weight)
    return out.view(B, S, HIDDEN)
