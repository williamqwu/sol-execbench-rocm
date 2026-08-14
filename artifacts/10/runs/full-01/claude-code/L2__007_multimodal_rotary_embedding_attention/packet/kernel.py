import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem constants (from definition.json)
# ---------------------------------------------------------------------------
NUM_HEADS = 28
NUM_KV_HEADS = 4
NUM_KV_GROUPS = 7
HEAD_DIM = 128
HIDDEN = 3584
Q_DIM = NUM_HEADS * HEAD_DIM        # 3584
KV_DIM = NUM_KV_HEADS * HEAD_DIM    # 512
QKV_DIM = Q_DIM + 2 * KV_DIM        # 4608
K_OFF = Q_DIM                       # 3584
V_OFF = Q_DIM + KV_DIM              # 4096
N_ROPE_HEADS = NUM_HEADS + NUM_KV_HEADS  # 32
SCALING = HEAD_DIM ** -0.5

# constexpr mirrors usable inside @triton.jit
_NH = tl.constexpr(NUM_HEADS)
_NKVG = tl.constexpr(NUM_KV_GROUPS)
_HD = tl.constexpr(HEAD_DIM)
_KOFF = tl.constexpr(K_OFF)
_VOFF = tl.constexpr(V_OFF)
_QKVD = tl.constexpr(QKV_DIM)
_QD = tl.constexpr(Q_DIM)


# ---------------------------------------------------------------------------
# In-place multimodal (3D) RoPE over the Q and K regions of the fused QKV buffer
#
# cos_combined[..., d] = cos[sec(d), b, s, d] with
#   sec(d) = 0 for d < 32, 1 for 32 <= d < 80, 2 for d >= 80
# (mrope_section [16,24,24] doubled -> [32,48,48]; split i is taken from plane i%3)
# ---------------------------------------------------------------------------
@triton.jit
def _mrope_inplace(
    QKV,            # [BS, 4608] bf16
    COS, SIN,       # [3, B, S, 128] bf16
    BS,
    PLANE,          # B * S * 128 elements per rope plane
    BLOCK_S: tl.constexpr,
    NHEADS: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_s = offs_s < BS

    dl = tl.arange(0, 64)
    plane_lo = tl.where(dl < 32, 0, 1)          # global d = dl        (0..63)
    plane_hi = tl.where(dl < 16, 1, 2)          # global d = 64 + dl   (64..127)

    rope_lo = offs_s[:, None] * 128 + dl[None, :]
    rope_hi = rope_lo + 64

    c_lo = tl.load(COS + plane_lo[None, :] * PLANE + rope_lo, mask=m_s[:, None], other=0.0).to(tl.float32)
    c_hi = tl.load(COS + plane_hi[None, :] * PLANE + rope_hi, mask=m_s[:, None], other=0.0).to(tl.float32)
    s_lo = tl.load(SIN + plane_lo[None, :] * PLANE + rope_lo, mask=m_s[:, None], other=0.0).to(tl.float32)
    s_hi = tl.load(SIN + plane_hi[None, :] * PLANE + rope_hi, mask=m_s[:, None], other=0.0).to(tl.float32)

    row = QKV + offs_s[:, None] * _QKVD + dl[None, :]
    for h in tl.static_range(NHEADS):
        base = row + h * 128
        x_lo = tl.load(base, mask=m_s[:, None], other=0.0).to(tl.float32)
        x_hi = tl.load(base + 64, mask=m_s[:, None], other=0.0).to(tl.float32)

        # the reference rounds every elementwise product to bf16 before adding
        t1 = (x_lo * c_lo).to(tl.bfloat16).to(tl.float32)
        t2 = ((-x_hi) * s_lo).to(tl.bfloat16).to(tl.float32)
        t3 = (x_hi * c_hi).to(tl.bfloat16).to(tl.float32)
        t4 = (x_lo * s_hi).to(tl.bfloat16).to(tl.float32)

        tl.store(base, (t1 + t2).to(tl.bfloat16), mask=m_s[:, None])
        tl.store(base + 64, (t3 + t4).to(tl.bfloat16), mask=m_s[:, None])


# ---------------------------------------------------------------------------
# Fused GQA attention, numerically identical to the reference.
#
# The reference computes  softmax(...) -> fp32, rounds the NORMALISED
# probability to bf16, then does the PV matmul. A single-pass online softmax
# rounds the UNNORMALISED probability instead, which is not the same rounding.
# So this runs two passes over K/V: pass 1 gets (m, l), pass 2 recomputes the
# scores and rounds p/l to bf16 exactly as the reference does.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd(
    QKV,            # [B, S, 4608] bf16
    MASK,           # [B, 1, S, S] bf16
    OUT,            # [B, S, 3584] bf16
    S,
    stride_qkv_b,
    stride_mask_b,
    stride_out_b,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // _NH
    h = pid_bh % _NH
    kvh = h // _NKVG

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, _HD)
    m_m = offs_m < S

    qkv_b = QKV + b * stride_qkv_b
    q = tl.load(
        qkv_b + offs_m[:, None] * _QKVD + h * _HD + offs_d[None, :],
        mask=m_m[:, None], other=0.0,
    )

    mask_b = MASK + b * stride_mask_b + offs_m[:, None] * S
    k_base = qkv_b + _KOFF + kvh * _HD + offs_d[None, :]
    v_base = qkv_b + _VOFF + kvh * _HD + offs_d[None, :]

    # ---- pass 1: row max and row sum of exp ------------------------------
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_m = offs_n < S
        if EVEN_N:
            k = tl.load(k_base + offs_n[:, None] * _QKVD)
            mk = tl.load(mask_b + offs_n[None, :], mask=m_m[:, None], other=0.0)
        else:
            k = tl.load(k_base + offs_n[:, None] * _QKVD, mask=n_m[:, None], other=0.0)
            mk = tl.load(mask_b + offs_n[None, :], mask=m_m[:, None] & n_m[None, :], other=0.0)

        # bf16 x bf16 -> fp32 accumulate, then torch rounds the result to bf16
        qk = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = (qk + mk.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        if not EVEN_N:
            qk = tl.where(n_m[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(qk - m_new[:, None]), 1)
        m_i = m_new

    # ---- pass 2: exact normalised probabilities, then PV -----------------
    acc = tl.zeros([BLOCK_M, _HD], dtype=tl.float32)
    inv_l = 1.0 / l_i

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_m = offs_n < S
        if EVEN_N:
            k = tl.load(k_base + offs_n[:, None] * _QKVD)
            mk = tl.load(mask_b + offs_n[None, :], mask=m_m[:, None], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * _QKVD)
        else:
            k = tl.load(k_base + offs_n[:, None] * _QKVD, mask=n_m[:, None], other=0.0)
            mk = tl.load(mask_b + offs_n[None, :], mask=m_m[:, None] & n_m[None, :], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * _QKVD, mask=n_m[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = (qk + mk.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

        p = tl.exp(qk - m_i[:, None]) * inv_l[:, None]
        if not EVEN_N:
            p = tl.where(n_m[None, :], p, 0.0)

        acc = tl.dot(p.to(tl.bfloat16), v, acc)

    tl.store(
        OUT + b * stride_out_b + offs_m[:, None] * _QD + h * _HD + offs_d[None, :],
        acc.to(tl.bfloat16),
        mask=m_m[:, None],
    )


def _attn_config(bsz, q_len):
    """Pick BLOCK_M so the grid fills the 256 CUs without wasting tiles."""
    best = None
    for bm in (128, 64, 32):
        if bm > 128:
            continue
        tiles = triton.cdiv(q_len, bm)
        wgs = bsz * NUM_HEADS * tiles
        # prefer the largest tile that still gives us at least one full wave
        if wgs >= 256 or bm == 32:
            best = bm
            break
    if best is None:
        best = 32
    bn = 64 if q_len >= 64 else 32
    nw = 8 if best >= 128 else 4
    return best, bn, nw


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
):
    bsz, q_len, _ = hidden_states.shape
    BS = bsz * q_len
    dev = hidden_states.device

    h2d = hidden_states.reshape(BS, HIDDEN)

    # --- fused QKV projection into one [BS, 4608] buffer -------------------
    qkv = torch.empty((BS, QKV_DIM), dtype=torch.bfloat16, device=dev)
    torch.addmm(q_bias, h2d, q_weight.t(), out=qkv[:, :K_OFF])
    torch.addmm(k_bias, h2d, k_weight.t(), out=qkv[:, K_OFF:V_OFF])
    torch.addmm(v_bias, h2d, v_weight.t(), out=qkv[:, V_OFF:])

    # --- in-place multimodal RoPE on Q and K -------------------------------
    BLOCK_S = 8
    _mrope_inplace[(triton.cdiv(BS, BLOCK_S),)](
        qkv, cos, sin, BS, BS * HEAD_DIM,
        BLOCK_S=BLOCK_S, NHEADS=N_ROPE_HEADS,
        num_warps=4, num_stages=1,
    )

    # --- attention ----------------------------------------------------------
    attn_out = torch.empty((BS, Q_DIM), dtype=torch.bfloat16, device=dev)
    BLOCK_M, BLOCK_N, nw = _attn_config(bsz, q_len)
    _attn_fwd[(bsz * NUM_HEADS, triton.cdiv(q_len, BLOCK_M))](
        qkv, attention_mask, attn_out,
        q_len,
        q_len * QKV_DIM,
        q_len * q_len,
        q_len * Q_DIM,
        sm_scale=SCALING,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_N=(q_len % BLOCK_N == 0),
        num_warps=nw, num_stages=1,
    )

    # --- output projection --------------------------------------------------
    out = torch.mm(attn_out, o_weight.t())
    return out.view(bsz, q_len, HIDDEN)
