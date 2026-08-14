import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem constants (from definition.json)
# ---------------------------------------------------------------------------
HIDDEN = 4096
NHEAD = 32
NKV = 8
HDIM = 128
ROPE_THETA = 10000.0
DIM_T, DIM_H, DIM_W = 42, 42, 44

# ---------------------------------------------------------------------------
# RoPE inverse-frequency tables.  These are pure architectural constants (they
# do not depend on any input tensor), computed once per device with exactly the
# same expression the reference uses, so the fp32 values are bit identical.
# ---------------------------------------------------------------------------
_TABLES = {}


def _tables(device: torch.device):
    key = device.index if device.index is not None else 0
    tab = _TABLES.get(key)
    if tab is not None:
        return tab

    def inv(dim):
        return 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

    inv_1d = inv(HDIM)          # 64 entries
    inv_t = inv(DIM_T)          # 21
    inv_h = inv(DIM_H)          # 21
    inv_w = inv(DIM_W)          # 22
    inv_k = torch.cat([inv_t, inv_h, inv_w])  # 64 entries, lane-major
    tab = (inv_1d.contiguous(), inv_k.contiguous())
    _TABLES[key] = tab
    return tab


# ---------------------------------------------------------------------------
# 1-D RoPE over the query projection, applied in place.
# Layout: q is (B*Lq, NHEAD*HDIM) contiguous.
# One program == one token; 64 lanes per head cover both halves of the pair.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_q(
    q_ptr,
    pos_ptr,
    inv_ptr,
    NH: tl.constexpr,
    HD: tl.constexpr,
    HALF: tl.constexpr,
):
    tok = tl.program_id(0)

    lane = tl.arange(0, NH * HALF)
    h = lane // HALF
    d = lane % HALF

    pos = tl.load(pos_ptr + tok).to(tl.float32)
    invf = tl.load(inv_ptr + d)
    ang = pos * invf
    c = tl.cos(ang).to(tl.bfloat16)
    s = tl.sin(ang).to(tl.bfloat16)

    base = tok.to(tl.int64) * (NH * HD) + h * HD + d
    p1 = q_ptr + base
    p2 = p1 + HALF

    x1 = tl.load(p1)
    x2 = tl.load(p2)

    o1 = (x1 * c).to(tl.bfloat16) + ((-x2) * s).to(tl.bfloat16)
    o2 = (x2 * c).to(tl.bfloat16) + (x1 * s).to(tl.bfloat16)

    tl.store(p1, o1.to(tl.bfloat16))
    tl.store(p2, o2.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# 3-D RoPE over the key projection, applied in place.
# head_dim 128 is split 42 / 42 / 44 -> halves of 21 / 21 / 22.
# lane m in [0,64):
#   m < 21          -> t group, i1 = m,      i2 = m + 21
#   21 <= m < 42    -> h group, i1 = m + 21, i2 = m + 42
#   42 <= m < 64    -> w group, i1 = m + 42, i2 = m + 64
# inv_k[m] already holds the right inverse frequency for lane m.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_k(
    k_ptr,
    thw_ptr,
    inv_ptr,
    NH: tl.constexpr,
    HD: tl.constexpr,
    HALF: tl.constexpr,
):
    tok = tl.program_id(0)

    lane = tl.arange(0, NH * HALF)
    h = lane // HALF
    m = lane % HALF

    tp = tl.load(thw_ptr + tok.to(tl.int64) * 3 + 0).to(tl.float32)
    hp = tl.load(thw_ptr + tok.to(tl.int64) * 3 + 1).to(tl.float32)
    wp = tl.load(thw_ptr + tok.to(tl.int64) * 3 + 2).to(tl.float32)

    pos = tl.where(m < 21, tp, tl.where(m < 42, hp, wp))
    i1 = tl.where(m < 21, m, tl.where(m < 42, m + 21, m + 42))
    off = tl.where(m < 42, 21, 22)

    invf = tl.load(inv_ptr + m)
    ang = pos * invf
    c = tl.cos(ang).to(tl.bfloat16)
    s = tl.sin(ang).to(tl.bfloat16)

    base = tok.to(tl.int64) * (NH * HD) + h * HD + i1
    p1 = k_ptr + base
    p2 = k_ptr + base + off

    x1 = tl.load(p1)
    x2 = tl.load(p2)

    o1 = (x1 * c).to(tl.bfloat16) + ((-x2) * s).to(tl.bfloat16)
    o2 = (x2 * c).to(tl.bfloat16) + (x1 * s).to(tl.bfloat16)

    tl.store(p1, o1.to(tl.bfloat16))
    tl.store(p2, o2.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Cross-attention, GQA, no mask.
#
# The reference's rounding schedule is part of the spec and is reproduced here
# exactly.  In particular:
#   s  = round_bf16( round_bf16(Q @ K^T) / sqrt(128) )   two separate roundings
#   p  = round_bf16( softmax_fp32(s) )                   probs land in bf16
#   o  = round_bf16( p @ V )                             fp32 acc, bf16 out
# A standard flash kernel keeps p in fp32 and is *more* accurate than the
# reference, which fails the tolerance.  So the softmax is normalised before
# the PV product, which needs l up front: pass 1 derives (m, l), pass 2
# rebuilds s and accumulates.  QK is recomputed; PV is not.
#
# q: (B*Lq, NHEAD*HDIM)   k/v: (B*Lv, NKV*HDIM)   out: (B*Lq, NHEAD*HDIM)
# ---------------------------------------------------------------------------
@triton.jit
def _flash(
    q_ptr, k_ptr, v_ptr, o_ptr,
    Lq, Lv,
    sm_scale,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
    REP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NH
    h = pid_bh % NH
    kvh = h // REP

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    offs_n = tl.arange(0, BN)

    q_rows = b.to(tl.int64) * Lq + offs_m
    m_mask = offs_m < Lq
    q_off = q_rows[:, None] * (NH * HD) + h * HD + offs_d[None, :]
    q = tl.load(q_ptr + q_off, mask=m_mask[:, None], other=0.0)

    k_base = k_ptr + b.to(tl.int64) * Lv * (NKV * HD) + kvh * HD
    v_base = v_ptr + b.to(tl.int64) * Lv * (NKV * HD) + kvh * HD

    # ---- pass 1: row max and row sum, online ----
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)

    for start in range(0, tl.cdiv(Lv, BN) * BN, BN):
        n = start + offs_n
        n_mask = n < Lv
        kt = tl.load(
            k_base + n[None, :] * (NKV * HD) + offs_d[:, None],
            mask=n_mask[None, :], other=0.0,
        )
        s = tl.dot(q, kt)
        s = s.to(tl.bfloat16).to(tl.float32)
        s = s * sm_scale
        s = s.to(tl.bfloat16).to(tl.float32)
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # rows fully masked out cannot occur (Lv >= 1); guard anyway
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)

    # ---- pass 2: normalised probs, rounded to bf16, times V ----
    acc = tl.zeros([BM, HD], tl.float32)

    for start in range(0, tl.cdiv(Lv, BN) * BN, BN):
        n = start + offs_n
        n_mask = n < Lv
        kt = tl.load(
            k_base + n[None, :] * (NKV * HD) + offs_d[:, None],
            mask=n_mask[None, :], other=0.0,
        )
        s = tl.dot(q, kt)
        s = s.to(tl.bfloat16).to(tl.float32)
        s = s * sm_scale
        s = s.to(tl.bfloat16).to(tl.float32)
        s = tl.where(n_mask[None, :], s, float("-inf"))

        p = tl.exp(s - m_i[:, None]) / l_safe[:, None]
        p = p.to(tl.bfloat16)

        v = tl.load(
            v_base + n[:, None] * (NKV * HD) + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        acc = tl.dot(p, v, acc)

    o_off = q_rows[:, None] * (NH * HD) + h * HD + offs_d[None, :]
    tl.store(o_ptr + o_off, acc.to(tl.bfloat16), mask=m_mask[:, None])


def _attention(q, k, v, B, Lq, Lv):
    """q: (B*Lq, NHEAD*HDIM)  k/v: (B*Lv, NKV*HDIM) -> (B*Lq, NHEAD*HDIM)"""
    out = torch.empty_like(q)

    # Enough programs to fill 256 CUs: shrink the M tile when Lq is short.
    bm = 128
    while bm > 16 and B * NHEAD * triton.cdiv(Lq, bm) < 512:
        bm //= 2
    bn = 64
    nw = 4 if bm >= 64 else 2

    _flash[(triton.cdiv(Lq, bm), B * NHEAD)](
        q, k, v, out,
        Lq, Lv,
        1.0 / math.sqrt(HDIM),
        NHEAD, NKV, HDIM, NHEAD // NKV,
        bm, bn,
        num_warps=nw, num_stages=2,
    )
    return out


@torch.no_grad()
def run(
    language_hidden_states: torch.Tensor,
    vision_hidden_states: torch.Tensor,
    language_position_ids: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    B, Lq, _ = language_hidden_states.shape
    Lv = vision_hidden_states.shape[1]
    dev = language_hidden_states.device
    inv_1d, inv_k = _tables(dev)

    lang2 = language_hidden_states.reshape(B * Lq, HIDDEN)
    vis2 = vision_hidden_states.reshape(B * Lv, HIDDEN)

    # projections
    q = torch.addmm(q_proj_bias, lang2, q_proj_weight.t())
    k = torch.addmm(k_proj_bias, vis2, k_proj_weight.t())
    v = torch.addmm(v_proj_bias, vis2, v_proj_weight.t())

    pos = language_position_ids
    if not pos.is_contiguous():
        pos = pos.contiguous()
    thw = vision_grid_thw
    if not thw.is_contiguous():
        thw = thw.contiguous()

    _rope_q[(B * Lq,)](q, pos, inv_1d, NHEAD, HDIM, HDIM // 2, num_warps=4)
    _rope_k[(B * Lv,)](k, thw, inv_k, NKV, HDIM, HDIM // 2, num_warps=2)

    # attention writes straight into (B*Lq, NHEAD*HDIM) layout, which is
    # already what the output projection wants -- no transpose, no copy.
    o = _attention(q, k, v, B, Lq, Lv)
    out = torch.mm(o, o_proj_weight.t())
    return out.view(B, Lq, HIDDEN)
