import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NEG = tl.constexpr(-1.0e30)


@triton.jit
def _kv_norm_rope(K, KO, COS, SIN, W, eps,
                  NH: tl.constexpr, D: tl.constexpr):
    """RMSNorm + RoPE over K, laid out [B*S, NH, D] -> same, bf16."""
    t = tl.program_id(0)
    offs_h = tl.arange(0, NH)
    offs_d = tl.arange(0, D)
    sw = (offs_d + D // 2) % D
    sign = tl.where(offs_d < D // 2, -1.0, 1.0)

    base = t * (NH * D)
    p = base + offs_h[:, None] * D + offs_d[None, :]
    ps = base + offs_h[:, None] * D + sw[None, :]

    x = tl.load(K + p).to(tl.float32)
    xs = tl.load(K + ps).to(tl.float32)
    w = tl.load(W + offs_d).to(tl.float32)
    ws = tl.load(W + sw).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    xn = (w[None, :] * (x * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
    xsn = (ws[None, :] * (xs * rstd[:, None])).to(tl.bfloat16).to(tl.float32) * sign[None, :]

    cp = t * D + offs_d
    c = tl.load(COS + cp).to(tl.float32)
    sn = tl.load(SIN + cp).to(tl.float32)

    o = (xn * c[None, :]).to(tl.bfloat16).to(tl.float32) + \
        (xsn * sn[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(KO + p, o.to(tl.bfloat16))


@triton.jit
def _attn_fwd(Q, K, V, O, COS, SIN, QW, S, sm_scale, eps,
              H: tl.constexpr, G: tl.constexpr, NKV: tl.constexpr,
              D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              EVEN_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hkv = h // G

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    sw = (offs_d + D // 2) % D
    sign = tl.where(offs_d < D // 2, -1.0, 1.0)

    # ---- load Q tile, fused RMSNorm + RoPE (exact reference rounding) ----
    qbase = (b * S) * (H * D) + h * D
    qp = qbase + offs_m[:, None] * (H * D) + offs_d[None, :]
    qps = qbase + offs_m[:, None] * (H * D) + sw[None, :]
    if EVEN_M:
        x = tl.load(Q + qp).to(tl.float32)
        xs = tl.load(Q + qps).to(tl.float32)
    else:
        mrow = offs_m < S
        x = tl.load(Q + qp, mask=mrow[:, None], other=0.0).to(tl.float32)
        xs = tl.load(Q + qps, mask=mrow[:, None], other=0.0).to(tl.float32)
    w = tl.load(QW + offs_d).to(tl.float32)
    ws = tl.load(QW + sw).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    xn = (w[None, :] * (x * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
    xsn = (ws[None, :] * (xs * rstd[:, None])).to(tl.bfloat16).to(tl.float32) * sign[None, :]

    cp = (b * S) * D + offs_m[:, None] * D + offs_d[None, :]
    if EVEN_M:
        c = tl.load(COS + cp).to(tl.float32)
        sn = tl.load(SIN + cp).to(tl.float32)
    else:
        c = tl.load(COS + cp, mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
        sn = tl.load(SIN + cp, mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
    q = ((xn * c).to(tl.bfloat16).to(tl.float32) +
         (xsn * sn).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    kvbase = (b * S) * (NKV * D) + hkv * D
    hi = tl.minimum((pid_m + 1) * BLOCK_M, S)
    nfull = (hi // BLOCK_N) * BLOCK_N

    # ================= pass 1: row max + row sum (fp32 softmax stats) =========
    m_i = tl.full([BLOCK_M], NEG, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    for start_n in range(0, nfull, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kp = kvbase + offs_n[:, None] * (NKV * D) + offs_d[None, :]
        k = tl.load(K + kp)
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, NEG)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(qk - m_new[:, None]), 1)
        m_i = m_new

    for start_n in range(nfull, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        kp = kvbase + offs_n[:, None] * (NKV * D) + offs_d[None, :]
        k = tl.load(K + kp, mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = tl.where((offs_m[:, None] >= offs_n[None, :]) & nmask[None, :], qk, NEG)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(qk - m_new[:, None]), 1)
        m_i = m_new


    # ============ pass 2: normalized p -> bf16 -> PV (fp32 accumulate) ========
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, nfull, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kp = kvbase + offs_n[:, None] * (NKV * D) + offs_d[None, :]
        k = tl.load(K + kp)
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, NEG)
        p = (tl.exp(qk - m_i[:, None]) / l_i[:, None]).to(tl.bfloat16)
        v = tl.load(V + kp)
        acc = tl.dot(p, v, acc)

    for start_n in range(nfull, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        kp = kvbase + offs_n[:, None] * (NKV * D) + offs_d[None, :]
        k = tl.load(K + kp, mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.bfloat16).to(tl.float32)
        qk = (qk * sm_scale).to(tl.bfloat16).to(tl.float32)
        qk = tl.where((offs_m[:, None] >= offs_n[None, :]) & nmask[None, :], qk, NEG)
        p = (tl.exp(qk - m_i[:, None]) / l_i[:, None]).to(tl.bfloat16)
        v = tl.load(V + kp, mask=nmask[:, None], other=0.0)
        acc = tl.dot(p, v, acc)

    op = (b * S) * (H * D) + offs_m[:, None] * (H * D) + h * D + offs_d[None, :]
    if EVEN_M:
        tl.store(O + op, acc.to(tl.bfloat16))
    else:
        tl.store(O + op, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)


NUM_HEADS = 96
NUM_KV = 8
HEAD_DIM = 128
GROUPS = 12


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    B, S, _ = hidden_states.shape
    T = B * S
    hs = hidden_states.reshape(T, -1)

    q = F.linear(hs, q_proj_weight, q_proj_bias)
    k = F.linear(hs, k_proj_weight, k_proj_bias)
    v = F.linear(hs, v_proj_weight, v_proj_bias)

    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    kn = torch.empty_like(k)
    _kv_norm_rope[(T,)](
        k, kn, cos_c, sin_c, k_norm_weight, rms_norm_eps,
        NH=NUM_KV, D=HEAD_DIM, num_warps=4, num_stages=1,
    )

    attn = torch.empty_like(q)
    if S <= 128:
        BM, BN, nw = 64, 64, 4
    else:
        BM, BN, nw = 128, 64, 4
    grid = (triton.cdiv(S, BM), B * NUM_HEADS)
    _attn_fwd[grid](
        q, kn, v, attn, cos_c, sin_c, q_norm_weight, S,
        HEAD_DIM ** -0.5, rms_norm_eps,
        H=NUM_HEADS, G=GROUPS, NKV=NUM_KV, D=HEAD_DIM,
        BLOCK_M=BM, BLOCK_N=BN, EVEN_M=(S % BM == 0),
        num_warps=nw, num_stages=2,
    )

    out = F.linear(attn, o_proj_weight, None)
    return out.view(B, S, -1)
