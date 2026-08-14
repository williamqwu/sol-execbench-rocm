import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



# ---------------------------------------------------------------------------
# Fused RMSNorm(1+w) + RoPE for Q and K, in the [B, S, H, D] layout.
#
# Faithfully reproduces the reference's intermediate bfloat16 roundings:
#   xn  = bf16( x_f32 * rsqrt(mean(x^2)+eps) * (1 + w_f32) )
#   out = bf16( bf16(xn*cos) + bf16(rotate_half(xn)*sin) )
# ---------------------------------------------------------------------------
@triton.jit
def _nr_body(IP, OP, W, cs, sn, bs, hoff, H: tl.constexpr, D: tl.constexpr,
             GH: tl.constexpr, eps):
    d = tl.arange(0, D)
    perm = (d + (D // 2)) % D
    sign = tl.where(d < (D // 2), -1.0, 1.0)
    hs = hoff + tl.arange(0, GH)

    base = IP + bs.to(tl.int64) * (H * D) + hs[:, None].to(tl.int64) * D
    w = tl.load(W + d).to(tl.float32)
    wp = tl.load(W + perm).to(tl.float32)

    x = tl.load(base + d[None, :]).to(tl.float32)
    xp = tl.load(base + perm[None, :]).to(tl.float32)

    var = tl.sum(x * x, axis=1) * (1.0 / D)
    r = tl.rsqrt(var + eps)

    xn = (x * r[:, None] * (1.0 + w[None, :])).to(tl.bfloat16).to(tl.float32)
    xpn = (xp * r[:, None] * (1.0 + wp[None, :])).to(tl.bfloat16).to(tl.float32)

    p1 = (xn * cs[None, :]).to(tl.bfloat16).to(tl.float32)
    p2 = ((sign[None, :] * xpn) * sn[None, :]).to(tl.bfloat16).to(tl.float32)
    res = (p1 + p2).to(tl.bfloat16)

    out = OP + bs.to(tl.int64) * (H * D) + hs[:, None].to(tl.int64) * D
    tl.store(out + d[None, :], res)


@triton.jit
def _norm_rope(QP, KP, COS, SIN, QW, KW, QO, KO, eps,
               HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
               GH: tl.constexpr, NQG: tl.constexpr):
    bs = tl.program_id(0)
    g = tl.program_id(1)
    d = tl.arange(0, D)
    cs = tl.load(COS + bs.to(tl.int64) * D + d).to(tl.float32)
    sn = tl.load(SIN + bs.to(tl.int64) * D + d).to(tl.float32)
    if g < NQG:
        _nr_body(QP, QO, QW, cs, sn, bs, g * GH, HQ, D, GH, eps)
    else:
        _nr_body(KP, KO, KW, cs, sn, bs, (g - NQG) * GH, HK, D, GH, eps)


# ---------------------------------------------------------------------------
# Fused attention:  softcap(tanh) -> + mask -> softmax(fp32) -> @V
# Reproduces the reference's bf16 rounding of the logit chain.
# ---------------------------------------------------------------------------
@triton.jit
def _logits(q, k, mk, scaling, softcap):
    """QK^T -> *scaling -> /cap -> tanh -> *cap -> +mask, with the reference's
    bfloat16 rounding after every step."""
    s = tl.dot(q, tl.trans(k))
    s = s.to(tl.bfloat16).to(tl.float32)
    s = (s * scaling).to(tl.bfloat16).to(tl.float32)
    s = (s / softcap).to(tl.bfloat16).to(tl.float32)
    s = libdevice.tanh(s).to(tl.bfloat16).to(tl.float32)
    s = (s * softcap).to(tl.bfloat16).to(tl.float32)
    s = (s + mk).to(tl.bfloat16).to(tl.float32)
    return s


@triton.jit
def _attn_fwd(Q, K, V, M, O, S, scaling, softcap,
              HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
              NM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    # heads vary fastest: the 24 q-heads of one (b, m-block) run together and
    # share the same BM x S slice of the mask in L2.
    h = pid % HQ
    t = pid // HQ
    mblk = t % NM
    b = t // NM
    kh = h // (HQ // HK)

    offs_m = mblk * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mmask = offs_m < S

    qb = Q + b.to(tl.int64) * (S * HQ * D) + h.to(tl.int64) * D
    q = tl.load(qb + offs_m[:, None].to(tl.int64) * (HQ * D) + offs_d[None, :],
                mask=mmask[:, None], other=0.0)

    kb = K + b.to(tl.int64) * (S * HK * D) + kh.to(tl.int64) * D
    vb = V + b.to(tl.int64) * (S * HK * D) + kh.to(tl.int64) * D
    mb = M + b.to(tl.int64) * (S * S) + offs_m[:, None].to(tl.int64) * S

    # ---- pass 1: row max and row sum -------------------------------------
    # The reference rounds the *normalized* softmax to bfloat16, so the row sum
    # must be known before any rounding happens. That forces two passes.
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)

    for start in range(0, S, BN):
        offs_n = start + tl.arange(0, BN)
        if EVEN_N:
            k = tl.load(kb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :])
            mk = tl.load(mb + offs_n[None, :], mask=mmask[:, None], other=0.0)
            s = _logits(q, k, mk.to(tl.float32), scaling, softcap)
        else:
            nm = offs_n < S
            k = tl.load(kb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :],
                        mask=nm[:, None], other=0.0)
            mk = tl.load(mb + offs_n[None, :],
                         mask=mmask[:, None] & nm[None, :], other=0.0)
            s = _logits(q, k, mk.to(tl.float32), scaling, softcap)
            s = tl.where(nm[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(s - m_new[:, None]), 1)
        m_i = m_new

    r_i = 1.0 / l_i

    # ---- pass 2: normalized probabilities, rounded to bf16, times V -------
    acc = tl.zeros([BM, D], tl.float32)

    for start in range(0, S, BN):
        offs_n = start + tl.arange(0, BN)
        if EVEN_N:
            k = tl.load(kb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :])
            v = tl.load(vb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :])
            mk = tl.load(mb + offs_n[None, :], mask=mmask[:, None], other=0.0)
            s = _logits(q, k, mk.to(tl.float32), scaling, softcap)
            p = tl.exp(s - m_i[:, None]) * r_i[:, None]
        else:
            nm = offs_n < S
            k = tl.load(kb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :],
                        mask=nm[:, None], other=0.0)
            v = tl.load(vb + offs_n[:, None].to(tl.int64) * (HK * D) + offs_d[None, :],
                        mask=nm[:, None], other=0.0)
            mk = tl.load(mb + offs_n[None, :],
                         mask=mmask[:, None] & nm[None, :], other=0.0)
            s = _logits(q, k, mk.to(tl.float32), scaling, softcap)
            p = tl.exp(s - m_i[:, None]) * r_i[:, None]
            p = tl.where(nm[None, :], p, 0.0)

        acc += tl.dot(p.to(tl.bfloat16), v)

    ob = O + b.to(tl.int64) * (S * HQ * D) + h.to(tl.int64) * D
    tl.store(ob + offs_m[:, None].to(tl.int64) * (HQ * D) + offs_d[None, :],
             acc.to(tl.bfloat16), mask=mmask[:, None])


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    B, S, HS = hidden_states.shape
    HQ = 24
    HK = 8
    D = 128
    scaling = D ** -0.5
    softcap = float(attn_logit_softcapping)
    eps = float(rms_norm_eps)

    hs2 = hidden_states.reshape(B * S, HS)

    q = torch.mm(hs2, q_proj_weight.t())
    k = torch.mm(hs2, k_proj_weight.t())
    v = torch.mm(hs2, v_proj_weight.t())

    qo = torch.empty_like(q)
    ko = torch.empty_like(k)

    GH = 4
    NQG = HQ // GH
    _norm_rope[(B * S, NQG + HK // GH)](
        q, k, cos, sin, q_norm_weight, k_norm_weight, qo, ko, eps,
        HQ=HQ, HK=HK, D=D, GH=GH, NQG=NQG,
        num_warps=2, num_stages=1,
    )

    out = torch.empty_like(q)

    if S <= 256:
        BM, BN = 64, 64
    else:
        BM, BN = 128, 64
    NM = triton.cdiv(S, BM)
    even_n = (S % BN) == 0

    _attn_fwd[(NM * B * HQ,)](
        qo, ko, v, attention_mask, out, S, scaling, softcap,
        HQ=HQ, HK=HK, D=D, NM=NM, BM=BM, BN=BN, EVEN_N=even_n,
        num_warps=4, num_stages=1,
    )

    return torch.mm(out, o_proj_weight.t()).view(B, S, HS)
