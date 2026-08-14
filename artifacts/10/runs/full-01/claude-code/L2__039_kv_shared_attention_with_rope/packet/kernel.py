import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: RMSNorm (+RoPE for K) applied to the single KV head.
#   Kraw/Vraw : [T, D]
#   Kout/Vout : [T, D]   (== [B, 1, S, D])
# ---------------------------------------------------------------------------
@triton.jit
def _kv_norm_rope(
    Kraw, Vraw, Kout, Vout, KW, Pos, InvF,
    S, eps,
    stride_pb, stride_ps,
    D: tl.constexpr, HD: tl.constexpr,
):
    t = tl.program_id(0)
    b = t // S
    s = t - b * S

    i = tl.arange(0, HD)
    p = tl.load(Pos + b * stride_pb + s * stride_ps).to(tl.float32)
    ang = p * tl.load(InvF + i)
    cf = tl.cos(ang).to(tl.bfloat16).to(tl.float32)
    sf = tl.sin(ang).to(tl.bfloat16).to(tl.float32)

    w1 = tl.load(KW + i).to(tl.float32)
    w2 = tl.load(KW + HD + i).to(tl.float32)

    o = t * D

    # ---- K: rmsnorm -> weight -> bf16 -> rope ----
    x1 = tl.load(Kraw + o + i).to(tl.float32)
    x2 = tl.load(Kraw + o + HD + i).to(tl.float32)
    var = (tl.sum(x1 * x1) + tl.sum(x2 * x2)) / D
    rs = 1.0 / tl.sqrt(var + eps)
    n1 = (x1 * rs * w1).to(tl.bfloat16).to(tl.float32)
    n2 = (x2 * rs * w2).to(tl.bfloat16).to(tl.float32)
    r1 = ((n1 * cf).to(tl.bfloat16).to(tl.float32)
          + ((-n2) * sf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    r2 = ((n2 * cf).to(tl.bfloat16).to(tl.float32)
          + (n1 * sf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(Kout + o + i, r1)
    tl.store(Kout + o + HD + i, r2)

    # ---- V: rmsnorm only ----
    y1 = tl.load(Vraw + o + i).to(tl.float32)
    y2 = tl.load(Vraw + o + HD + i).to(tl.float32)
    var2 = (tl.sum(y1 * y1) + tl.sum(y2 * y2)) / D
    rs2 = 1.0 / tl.sqrt(var2 + eps)
    tl.store(Vout + o + i, (y1 * rs2).to(tl.bfloat16))
    tl.store(Vout + o + HD + i, (y2 * rs2).to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Kernel B: (Q rmsnorm + rope) fused with flash attention, tanh soft-capping
#           and an arbitrary additive mask.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd(
    Qraw, K, V, M, O, QW, Pos, InvF,
    S, softcap, eps,
    stride_pb, stride_ps, stride_mb, stride_ms,
    NH: tl.constexpr, D: tl.constexpr, HD: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, QSCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NH
    h = bh - b * NH

    offm = pid_m * BM + tl.arange(0, BM)
    rm = offm < S
    i = tl.arange(0, HD)

    row = (b * S + offm)[:, None] * (NH * D) + h * D
    q1 = tl.load(Qraw + row + i[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    q2 = tl.load(Qraw + row + (HD + i)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)

    var = (tl.sum(q1 * q1, 1) + tl.sum(q2 * q2, 1)) / D
    rs = (1.0 / tl.sqrt(var + eps))[:, None]
    w1 = tl.load(QW + i).to(tl.float32)[None, :]
    w2 = tl.load(QW + HD + i).to(tl.float32)[None, :]
    n1 = (q1 * rs * w1).to(tl.bfloat16).to(tl.float32)
    n2 = (q2 * rs * w2).to(tl.bfloat16).to(tl.float32)

    pos = tl.load(Pos + b * stride_pb + offm * stride_ps, mask=rm, other=0).to(tl.float32)[:, None]
    ang = pos * tl.load(InvF + i)[None, :]
    cf = tl.cos(ang).to(tl.bfloat16).to(tl.float32)
    sf = tl.sin(ang).to(tl.bfloat16).to(tl.float32)

    Q1 = ((n1 * cf).to(tl.bfloat16).to(tl.float32)
          + ((-n2) * sf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    Q2 = ((n2 * cf).to(tl.bfloat16).to(tl.float32)
          + (n1 * sf).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    acc1 = tl.zeros([BM, HD], dtype=tl.float32)
    acc2 = tl.zeros([BM, HD], dtype=tl.float32)
    m_i = tl.full([BM], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)

    kbase = b * S * D
    mbase = b * stride_mb + offm[:, None] * stride_ms
    NEG = float("-inf")
    hi = tl.cdiv(S, BN)

    for blk in range(0, hi):
        offn = blk * BN + tl.arange(0, BN)
        rn = offn < S
        mt = tl.load(M + mbase + offn[None, :],
                     mask=rm[:, None] & rn[None, :], other=NEG).to(tl.float32)
        if tl.max(mt) > NEG:
            kr = kbase + offn[:, None] * D
            k1 = tl.load(K + kr + i[None, :], mask=rn[:, None], other=0.0)
            k2 = tl.load(K + kr + (HD + i)[None, :], mask=rn[:, None], other=0.0)

            s = tl.dot(Q1, tl.trans(k1)) + tl.dot(Q2, tl.trans(k2))
            # reproduce the reference's intermediate bf16 roundings
            s = s.to(tl.bfloat16).to(tl.float32) * QSCALE
            s = (s / softcap).to(tl.bfloat16).to(tl.float32)
            e = tl.exp(-2.0 * tl.abs(s))
            th = (1.0 - e) / (1.0 + e)
            th = tl.where(s >= 0.0, th, -th)
            s = (th.to(tl.bfloat16).to(tl.float32) * softcap).to(tl.bfloat16).to(tl.float32)
            s = (s + mt).to(tl.bfloat16).to(tl.float32)

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            pw = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pw, 1)
            pb = pw.to(tl.bfloat16)

            v1 = tl.load(V + kr + i[None, :], mask=rn[:, None], other=0.0)
            v2 = tl.load(V + kr + (HD + i)[None, :], mask=rn[:, None], other=0.0)
            acc1 = acc1 * alpha[:, None] + tl.dot(pb, v1)
            acc2 = acc2 * alpha[:, None] + tl.dot(pb, v2)
            m_i = m_new

    li = l_i[:, None]
    tl.store(O + row + i[None, :], (acc1 / li).to(tl.bfloat16), mask=rm[:, None])
    tl.store(O + row + (HD + i)[None, :], (acc2 / li).to(tl.bfloat16), mask=rm[:, None])


_INVF = {}


def _inv_freq(theta, device, D=256):
    key = (float(theta), str(device), D)
    v = _INVF.get(key)
    if v is None:
        v = 1.0 / (float(theta) ** (
            torch.arange(0, D, 2, dtype=torch.float32, device=device) / D))
        _INVF[key] = v
    return v


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
    rope_theta: float,
    softcap: float,
    rms_norm_eps: float,
):
    B, S, H = hidden_states.shape
    NH = 8
    D = 256
    HD = D // 2
    dev = hidden_states.device

    hs = hidden_states.reshape(B * S, H)

    qraw = torch.mm(hs, q_proj_weight.t())          # [T, NH*D]
    kraw = torch.mm(hs, k_proj_weight.t())          # [T, D]
    vraw = torch.mm(hs, v_proj_weight.t())          # [T, D]

    key_out = torch.empty((B, 1, S, D), dtype=hidden_states.dtype, device=dev)
    val_out = torch.empty((B, 1, S, D), dtype=hidden_states.dtype, device=dev)

    invf = _inv_freq(rope_theta, dev, D)

    _kv_norm_rope[(B * S,)](
        kraw, vraw, key_out, val_out, k_norm_weight, position_ids, invf,
        S, float(rms_norm_eps),
        position_ids.stride(0), position_ids.stride(1),
        D=D, HD=HD, num_warps=2, num_stages=1,
    )

    out = torch.empty_like(qraw)
    BM = 64
    BN = 64
    grid = (triton.cdiv(S, BM), B * NH)
    _attn_fwd[grid](
        qraw, key_out, val_out, attention_mask, out, q_norm_weight,
        position_ids, invf,
        S, float(softcap), float(rms_norm_eps),
        position_ids.stride(0), position_ids.stride(1),
        attention_mask.stride(0), attention_mask.stride(2),
        NH=NH, D=D, HD=HD, BM=BM, BN=BN, QSCALE=1.0 / math.sqrt(D),
        num_warps=4, num_stages=1,
    )

    attn_output = torch.mm(out, o_proj_weight.t()).view(B, S, H)
    return attn_output, key_out, val_out
