import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- RMSNorm ---
@triton.jit
def _norm_apply(X, W, RS, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    """y = w * bf16(x * rstd), with rstd precomputed by torch (bit-exact)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    rstd = tl.load(RS + row)
    x = tl.load(X + row * N + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    xn = (x * rstd).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * N + cols, (w * xn).to(tl.bfloat16), mask=m)


# ------------------------------------------------------------------- RoPE ---
@triton.jit
def _rope(X, OUT, COS, SIN, HDIM: tl.constexpr, NH: tl.constexpr,
          HALF: tl.constexpr, BLOCK_D: tl.constexpr):
    m = tl.program_id(0)
    h = tl.arange(0, NH)[:, None]
    d = tl.arange(0, BLOCK_D)[None, :]
    dm = d < HALF
    base = X + m * (NH * HDIM) + h * HDIM + d
    x1 = tl.load(base, mask=dm, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=dm, other=0.0).to(tl.float32)
    c = tl.load(COS + m * HALF + d, mask=dm, other=0.0).to(tl.float32)
    s = tl.load(SIN + m * HALF + d, mask=dm, other=0.0).to(tl.float32)
    o1 = ((x1 * c).to(tl.bfloat16).to(tl.float32)
          - (x2 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((x1 * s).to(tl.bfloat16).to(tl.float32)
          + (x2 * c).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    ob = OUT + m * (NH * HDIM) + h * HDIM + d
    tl.store(ob, o1, mask=dm)
    tl.store(ob + HALF, o2, mask=dm)


# -------------------------------------------------------------- attention ---
@triton.jit
def _flash(Q, K, V, MSK, O, S, scale,
           NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    ph = tl.program_id(0)
    pid = tl.program_id(1)
    num_m = tl.cdiv(S, BLOCK_M)
    b = pid // num_m
    mb = pid % num_m
    hk = ph // (NH // NKV)

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < HD
    mrow = offs_m < S

    q_ptrs = Q + ((b * S + offs_m[:, None]) * NH + ph) * HD + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mrow[:, None] & dmask[None, :], other=0.0)

    mbase = MSK + (b * S + offs_m[:, None]) * S

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        ncol = offs_n < S
        k_ptrs = K + ((b * S + offs_n[:, None]) * NKV + hk) * HD + offs_d[None, :]
        k = tl.load(k_ptrs, mask=ncol[:, None] & dmask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = (s.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        mk = tl.load(mbase + offs_n[None, :],
                     mask=mrow[:, None] & ncol[None, :], other=0.0).to(tl.float32)
        s = (s + mk).to(tl.bfloat16).to(tl.float32)
        s = tl.where(ncol[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(s - m_new[:, None]), 1)
        m_i = m_new

    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    inv_l = 1.0 / l_i

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        ncol = offs_n < S
        k_ptrs = K + ((b * S + offs_n[:, None]) * NKV + hk) * HD + offs_d[None, :]
        k = tl.load(k_ptrs, mask=ncol[:, None] & dmask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = (s.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        mk = tl.load(mbase + offs_n[None, :],
                     mask=mrow[:, None] & ncol[None, :], other=0.0).to(tl.float32)
        s = (s + mk).to(tl.bfloat16).to(tl.float32)
        p = tl.exp(s - m_i[:, None]) * inv_l[:, None]
        p = tl.where(ncol[None, :], p, 0.0).to(tl.bfloat16)
        v_ptrs = V + ((b * S + offs_n[:, None]) * NKV + hk) * HD + offs_d[None, :]
        v = tl.load(v_ptrs, mask=ncol[:, None] & dmask[None, :], other=0.0)
        acc += tl.dot(p, v)

    o_ptrs = O + ((b * S + offs_m[:, None]) * NH + ph) * HD + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mrow[:, None] & dmask[None, :])


@triton.jit
def _flash_1blk(Q, K, V, MSK, O, S, scale,
                NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Whole key axis fits in one N-block (S <= BLOCK_N): compute QK^T once,
    softmax it in registers, and do PV. No recompute, exact reference order."""
    ph = tl.program_id(0)
    pid = tl.program_id(1)
    num_m = tl.cdiv(S, BLOCK_M)
    b = pid // num_m
    mb = pid % num_m
    hk = ph // (NH // NKV)

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < HD
    mrow = offs_m < S
    ncol = offs_n < S

    q = tl.load(Q + ((b * S + offs_m[:, None]) * NH + ph) * HD + offs_d[None, :],
                mask=mrow[:, None] & dmask[None, :], other=0.0)
    k = tl.load(K + ((b * S + offs_n[:, None]) * NKV + hk) * HD + offs_d[None, :],
                mask=ncol[:, None] & dmask[None, :], other=0.0)

    s = tl.dot(q, tl.trans(k))
    s = (s.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    mk = tl.load(MSK + (b * S + offs_m[:, None]) * S + offs_n[None, :],
                 mask=mrow[:, None] & ncol[None, :], other=0.0).to(tl.float32)
    s = (s + mk).to(tl.bfloat16).to(tl.float32)
    s = tl.where(ncol[None, :], s, -float("inf"))

    m_i = tl.max(s, 1)
    p = tl.exp(s - m_i[:, None])
    p = p / tl.sum(p, 1)[:, None]
    p = tl.where(ncol[None, :], p, 0.0).to(tl.bfloat16)

    v = tl.load(V + ((b * S + offs_n[:, None]) * NKV + hk) * HD + offs_d[None, :],
                mask=ncol[:, None] & dmask[None, :], other=0.0)
    acc = tl.dot(p, v)
    tl.store(O + ((b * S + offs_m[:, None]) * NH + ph) * HD + offs_d[None, :],
             acc.to(tl.bfloat16), mask=mrow[:, None] & dmask[None, :])


# ----------------------------------------------------------------- SwiGLU ---
@triton.jit
def _silu_mul(G, U, O, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(O + off, (s * u).to(tl.bfloat16), mask=m)


# ------------------------------------------------------------------- main ---
NUM_HEADS = 64
NUM_KV = 8
HEAD_DIM = 96
HALF_DIM = 48


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    B, Sq, H = hidden_states.shape
    M = B * Sq
    I = gate_proj_weight.shape[0]

    x = hidden_states.reshape(M, H)
    if not x.is_contiguous():
        x = x.contiguous()

    BLOCK_H = triton.next_power_of_2(H)

    # ---- pre-attention RMSNorm.
    # torch computes the fp32 row reduction + rsqrt (no Triton reduction order
    # reproduces it bitwise, and 1 ulp here is amplified ~1e5x downstream);
    # the elementwise scale is a Triton kernel, which is bit-exact and ~2x faster.
    xf = x.float()
    rstd1 = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + rms_norm_eps).view(-1)
    xn = torch.empty_like(x)
    _norm_apply[(M,)](x, input_layernorm_weight, rstd1, xn, H,
                      BLOCK=BLOCK_H, num_warps=8)

    # ---- QKV projections
    q = torch.mm(xn, q_proj_weight.t())
    k = torch.mm(xn, k_proj_weight.t())
    v = torch.mm(xn, v_proj_weight.t())

    # ---- RoPE
    cosf = cos.reshape(M, HALF_DIM)
    sinf = sin.reshape(M, HALF_DIM)
    if not cosf.is_contiguous():
        cosf = cosf.contiguous()
    if not sinf.is_contiguous():
        sinf = sinf.contiguous()
    qr = torch.empty_like(q)
    kr = torch.empty_like(k)
    _rope[(M,)](q, qr, cosf, sinf, HEAD_DIM, NUM_HEADS, HALF_DIM, 64, num_warps=4)
    _rope[(M,)](k, kr, cosf, sinf, HEAD_DIM, NUM_KV, HALF_DIM, 64, num_warps=1)

    # ---- attention
    attn = torch.empty_like(q)
    am = attention_mask
    if not am.is_contiguous():
        am = am.contiguous()
    # BLOCK_M=128 wins broadly; 64 only when Sq is not a multiple of 128 and
    # the tail block would be mostly wasted lanes.
    BLOCK_M = 128 if Sq % 128 == 0 else 64
    num_m = triton.cdiv(Sq, BLOCK_M)
    if Sq <= 256:
        # whole key axis in one block -> no QK recompute
        _flash_1blk[(NUM_HEADS, B * num_m)](
            qr, kr, v, am, attn, Sq, HEAD_DIM ** -0.5,
            NUM_HEADS, NUM_KV, HEAD_DIM,
            BLOCK_M=BLOCK_M, BLOCK_N=triton.next_power_of_2(Sq), BLOCK_D=128,
            num_warps=4, num_stages=1,
        )
    else:
        _flash[(NUM_HEADS, B * num_m)](
            qr, kr, v, am, attn, Sq, HEAD_DIM ** -0.5,
            NUM_HEADS, NUM_KV, HEAD_DIM,
            BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_D=128,
            num_warps=4, num_stages=1,
        )

    # ---- output projection + residual + post RMSNorm
    o = torch.mm(attn, o_proj_weight.t())
    h1 = x + o
    hf = h1.float()
    rstd2 = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + rms_norm_eps).view(-1)
    xn2 = torch.empty_like(x)
    _norm_apply[(M,)](h1, post_attention_layernorm_weight, rstd2, xn2, H,
                      BLOCK=BLOCK_H, num_warps=8)

    # ---- MLP
    g = torch.mm(xn2, gate_proj_weight.t())
    u = torch.mm(xn2, up_proj_weight.t())
    act = torch.empty_like(g)
    n = M * I
    _silu_mul[(triton.cdiv(n, 2048),)](g, u, act, n, BLOCK=2048, num_warps=4)
    d = torch.mm(act, down_proj_weight.t())

    out = h1 + d
    return out.view(B, Sq, H)
