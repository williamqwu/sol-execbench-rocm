import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused per-head RMSNorm + RoPE, done in place on the projection output.
#
# Reproduces the reference's exact rounding sequence:
#   xn  = bf16( x_f32 * rsqrt(mean(x^2) + eps) )
#   y   = bf16( xn * w )                        (bf16 * bf16 -> bf16)
#   out = bf16( bf16(y*cos) + bf16(rot(y)*sin) )
# ---------------------------------------------------------------------------
@triton.jit
def _norm_rope(
    X, COS, SIN, W,
    H: tl.constexpr, eps,
    BLOCK_H: tl.constexpr, D: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    ph = tl.program_id(1)

    offs_h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)

    base = tok * (H * D) + offs_h[:, None] * D
    ptr = X + base + offs_d[None, :]

    x = tl.load(ptr).to(tl.float32)
    var = tl.sum(x * x, axis=1) * (1.0 / D)
    rs = tl.math.rsqrt(var + eps)

    # rotate_half index map: [0..63] -> [64..127], [64..127] -> [0..63]
    half: tl.constexpr = D // 2
    idx_rot = tl.where(offs_d < half, offs_d + half, offs_d - half)
    xr = tl.load(X + base + idx_rot[None, :]).to(tl.float32)

    w = tl.load(W + offs_d).to(tl.float32)
    wr = tl.load(W + idx_rot).to(tl.float32)

    xn = (x * rs[:, None]).to(tl.bfloat16).to(tl.float32)
    xnr = (xr * rs[:, None]).to(tl.bfloat16).to(tl.float32)

    y = (xn * w[None, :]).to(tl.bfloat16).to(tl.float32)
    yr = (xnr * wr[None, :]).to(tl.bfloat16).to(tl.float32)
    yr = tl.where(offs_d[None, :] < half, -yr, yr)

    cs = tl.load(COS + tok * D + offs_d).to(tl.float32)
    sn = tl.load(SIN + tok * D + offs_d).to(tl.float32)

    a = (y * cs[None, :]).to(tl.bfloat16).to(tl.float32)
    b = (yr * sn[None, :]).to(tl.bfloat16).to(tl.float32)

    tl.store(ptr, (a + b).to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Attention with an arbitrary additive mask (B, 1, S, S) and GQA.
#
# The reference rounds the *normalized* softmax probabilities to bfloat16
# before the PV matmul.  The distribution is extremely peaked here, so that
# rounding is semantically load-bearing and a plain online-softmax flash
# kernel (which rounds the *unnormalized* weights) does not reproduce it.
#
# So: two passes over the KV axis inside one kernel launch.
#   pass 1 -> row max m and row sum l
#   pass 2 -> p = bf16(exp(s - m) / l), acc += p @ v
# Q stays in registers across both, and for a single KV block the second
# pass is skipped entirely (scores are still live).
#
# Score path matches the reference rounding:
#   s = bf16(q @ k^T) -> bf16(s * scaling) -> bf16(s + mask) -> fp32 softmax
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd(
    Q, K, V, M, O,
    S, scaling,
    sqb, sqs, skb, sks, smb, sob, sos,
    HQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, ONE_BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = (pid_bh // HQ).to(tl.int64)
    h = pid_bh % HQ
    kh = h // G

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    qp = Q + b * sqb + offs_m[:, None] * sqs + h * D + offs_d[None, :]
    if EVEN_M:
        q = tl.load(qp)
    else:
        q = tl.load(qp, mask=offs_m[:, None] < S, other=0.0)

    kbase = K + b * skb + kh * D
    vbase = V + b * skb + kh * D
    mbase = M + b * smb + offs_m[:, None] * S

    NEG: tl.constexpr = float("-inf")

    if ONE_BLOCK:
        offs_n = tl.arange(0, BLOCK_N)
        nm = offs_n < S
        if EVEN_N:
            k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :])
            mk = tl.load(mbase + offs_n[None, :]).to(tl.float32)
        else:
            k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :],
                        mask=nm[:, None], other=0.0)
            mk = tl.load(mbase + offs_n[None, :], mask=nm[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k))
        s = s.to(tl.bfloat16).to(tl.float32) * scaling
        s = s.to(tl.bfloat16).to(tl.float32)
        s = (s + mk).to(tl.bfloat16).to(tl.float32)
        if not EVEN_N:
            s = tl.where(nm[None, :], s, NEG)

        m_i = tl.max(s, 1)
        e = tl.exp(s - m_i[:, None])
        l_i = tl.sum(e, 1)
        p = (e / l_i[:, None]).to(tl.bfloat16)

        if EVEN_N:
            v = tl.load(vbase + offs_n[:, None] * sks + offs_d[None, :])
        else:
            v = tl.load(vbase + offs_n[:, None] * sks + offs_d[None, :],
                        mask=nm[:, None], other=0.0)
        acc = tl.dot(p, v)
    else:
        # ---- pass 1: row max / row sum ----
        m_i = tl.full([BLOCK_M], NEG, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if EVEN_N:
                k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :])
                mk = tl.load(mbase + offs_n[None, :]).to(tl.float32)
            else:
                nm = offs_n < S
                k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :],
                            mask=nm[:, None], other=0.0)
                mk = tl.load(mbase + offs_n[None, :], mask=nm[None, :], other=0.0).to(tl.float32)

            s = tl.dot(q, tl.trans(k))
            s = s.to(tl.bfloat16).to(tl.float32) * scaling
            s = s.to(tl.bfloat16).to(tl.float32)
            s = (s + mk).to(tl.bfloat16).to(tl.float32)
            if not EVEN_N:
                s = tl.where(offs_n[None, :] < S, s, NEG)

            m_new = tl.maximum(m_i, tl.max(s, 1))
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(s - m_new[:, None]), 1)
            m_i = m_new

        # ---- pass 2: normalized probabilities, PV ----
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if EVEN_N:
                k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :])
                mk = tl.load(mbase + offs_n[None, :]).to(tl.float32)
            else:
                nm = offs_n < S
                k = tl.load(kbase + offs_n[:, None] * sks + offs_d[None, :],
                            mask=nm[:, None], other=0.0)
                mk = tl.load(mbase + offs_n[None, :], mask=nm[None, :], other=0.0).to(tl.float32)

            s = tl.dot(q, tl.trans(k))
            s = s.to(tl.bfloat16).to(tl.float32) * scaling
            s = s.to(tl.bfloat16).to(tl.float32)
            s = (s + mk).to(tl.bfloat16).to(tl.float32)
            if not EVEN_N:
                s = tl.where(offs_n[None, :] < S, s, NEG)

            p = (tl.exp(s - m_i[:, None]) / l_i[:, None]).to(tl.bfloat16)

            if EVEN_N:
                v = tl.load(vbase + offs_n[:, None] * sks + offs_d[None, :])
            else:
                v = tl.load(vbase + offs_n[:, None] * sks + offs_d[None, :],
                            mask=offs_n[:, None] < S, other=0.0)
            acc += tl.dot(p, v)

    op = O + b * sob + offs_m[:, None] * sos + h * D + offs_d[None, :]
    if EVEN_M:
        tl.store(op, acc.to(tl.bfloat16))
    else:
        tl.store(op, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)


NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
GROUPS = 4


def _cfg(B, S):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages, waves)"""
    if S <= 128:
        return 128, 128, 4, 1
    if S <= 256:
        return 128, 256, 4, 1
    return 128, 32, 4, 2


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
    rms_norm_eps: float,
    scaling: float,
):
    B, S, Hd = hidden_states.shape
    n_tok = B * S

    hs = hidden_states.reshape(n_tok, Hd)
    q = torch.mm(hs, q_proj_weight.t())          # (n_tok, 4096)
    k = torch.mm(hs, k_proj_weight.t())          # (n_tok, 1024)
    v = torch.mm(hs, v_proj_weight.t())          # (n_tok, 1024)

    cosc = cos.reshape(n_tok, HEAD_DIM)
    sinc = sin.reshape(n_tok, HEAD_DIM)

    # ---- fused RMSNorm + RoPE (in place) ----
    _norm_rope[(n_tok, 2)](
        q, cosc, sinc, q_norm_weight, NUM_HEADS, rms_norm_eps,
        BLOCK_H=16, D=HEAD_DIM, num_warps=4, num_stages=1,
    )
    _norm_rope[(n_tok, 1)](
        k, cosc, sinc, k_norm_weight, NUM_KV_HEADS, rms_norm_eps,
        BLOCK_H=8, D=HEAD_DIM, num_warps=2, num_stages=1,
    )

    # ---- attention ----
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N, nw, ns = _cfg(B, S)

    _attn_fwd[(B * NUM_HEADS, triton.cdiv(S, BLOCK_M))](
        q, k, v, attention_mask, out,
        S, scaling,
        S * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM,
        S * NUM_KV_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM,
        S * S,
        S * NUM_HEADS * HEAD_DIM, NUM_HEADS * HEAD_DIM,
        HQ=NUM_HEADS, G=GROUPS, D=HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_M=(S % BLOCK_M == 0), EVEN_N=(S % BLOCK_N == 0),
        ONE_BLOCK=(S <= BLOCK_N),
        num_warps=nw, num_stages=ns,
    )

    o = torch.mm(out, o_proj_weight.t())
    return o.view(B, S, Hd)
