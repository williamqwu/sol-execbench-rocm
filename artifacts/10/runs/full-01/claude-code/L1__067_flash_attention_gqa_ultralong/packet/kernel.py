import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Model constants (fixed by the definition / reference).
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
GROUP = NUM_HEADS // NUM_KV_HEADS


@triton.jit
def _fmul(u, v):
    """Non-contractable fp32 multiply.

    The reference computes (x * cos) + (rot * sin) as two separately-rounded
    products followed by an add. Plain Triton contracts that into an FMA, which
    keeps extra precision in the product and diverges from the reference by
    ~1 ULP on 25% of elements. Emitting the multiply as inline asm forces the
    intermediate rounding the reference actually performs.
    """
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [u, v],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rope_k(
    K, COS, SIN,
    n_rows,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """In-place rotate-half RoPE on K, viewed as [B*S, HKV*D]."""
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = r < n_rows

    idx = tl.arange(0, (HKV * D) // 2)
    h = idx // (D // 2)
    j = idx % (D // 2)

    lo = h * D + j
    hi = lo + D // 2

    kp = K + r[:, None] * (HKV * D)
    cp = COS + r[:, None] * D
    sp = SIN + r[:, None] * D

    m2 = mask[:, None]
    klo = tl.load(kp + lo[None, :], mask=m2, other=0.0)
    khi = tl.load(kp + hi[None, :], mask=m2, other=0.0)
    cl = tl.load(cp + j[None, :], mask=m2, other=0.0)
    ch = tl.load(cp + (j + D // 2)[None, :], mask=m2, other=0.0)
    sl = tl.load(sp + j[None, :], mask=m2, other=0.0)
    sh = tl.load(sp + (j + D // 2)[None, :], mask=m2, other=0.0)

    tl.store(kp + lo[None, :], _fmul(klo, cl) - _fmul(khi, sl), mask=m2)
    tl.store(kp + hi[None, :], _fmul(khi, ch) + _fmul(klo, sh), mask=m2)


@triton.jit
def _attn_fwd(
    Q, K, V, COS, SIN, Out,
    S,
    sm_scale,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    GRP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HKV
    hkv = pid_bh % HKV

    stride_q = H * D
    stride_k = HKV * D

    q_base = Q + b * S * stride_q + hkv * GRP * D
    k_base = K + b * S * stride_k + hkv * D
    v_base = V + b * S * stride_k + hkv * D
    o_base = Out + b * S * stride_q + hkv * GRP * D
    cs_base = b * S * D

    # Tile rows stack the GRP query heads that share this KV head, so K/V is
    # loaded once per group instead of once per query head.
    rows = tl.arange(0, GRP * BLOCK_M)
    g = rows // BLOCK_M
    m = rows % BLOCK_M
    offs_m = pid_m * BLOCK_M + m
    mmask = offs_m < S

    offs_d = tl.arange(0, D)
    offs_n0 = tl.arange(0, BLOCK_N)

    # --- load Q tile (GRP heads at once) and apply RoPE in-register ---
    qrow = q_base + offs_m[:, None] * stride_q + g[:, None] * D
    d_rot = (offs_d + D // 2) % D

    q = tl.load(qrow + offs_d[None, :], mask=mmask[:, None], other=0.0)
    qr = tl.load(qrow + d_rot[None, :], mask=mmask[:, None], other=0.0)
    # rotate_half: negate the first half (exact sign flip)
    qr = tl.where(offs_d[None, :] < D // 2, -qr, qr)

    csp = cs_base + offs_m[:, None] * D + offs_d[None, :]
    cosv = tl.load(COS + csp, mask=mmask[:, None], other=0.0)
    sinv = tl.load(SIN + csp, mask=mmask[:, None], other=0.0)

    q = _fmul(q, cosv) + _fmul(qr, sinv)

    m_i = tl.full([GRP * BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GRP * BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([GRP * BLOCK_M, D], dtype=tl.float32)

    hi_n = tl.minimum((pid_m + 1) * BLOCK_M, S)
    n_full = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N

    for start_n in range(0, n_full, BLOCK_N):
        offs_n = start_n + offs_n0
        k = tl.load(k_base + offs_n[:, None] * stride_k + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * stride_k + offs_d[None, :])
        acc = tl.dot(p, v, acc, input_precision="ieee")
        m_i = m_new

    for start_n in range(n_full, hi_n, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < S
        k = tl.load(k_base + offs_n[:, None] * stride_k + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * stride_k + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        acc = tl.dot(p, v, acc, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    op = o_base + offs_m[:, None] * stride_q + g[:, None] * D + offs_d[None, :]
    tl.store(op, acc, mask=mmask[:, None])


def _cfg(B, S):
    # (BLOCK_M, BLOCK_N, num_warps, num_stages)
    if S <= 256:
        return 32, 64, 4, 1
    return 32, 128, 4, 1


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    H = NUM_HEADS
    HKV = NUM_KV_HEADS
    D = HEAD_DIM
    sm_scale = D ** -0.5

    B, S, _ = hidden_states.shape

    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)

    cos = cos.contiguous()
    sin = sin.contiguous()

    n_rows = B * S
    BLOCK_R = 4
    _rope_k[(triton.cdiv(n_rows, BLOCK_R),)](
        k, cos, sin, n_rows,
        HKV=HKV, D=D, BLOCK_R=BLOCK_R, num_warps=4,
    )

    out = torch.empty_like(q)

    BLOCK_M, BLOCK_N, nw, ns = _cfg(B, S)
    grid = (triton.cdiv(S, BLOCK_M), B * HKV)
    _attn_fwd[grid](
        q, k, v, cos, sin, out,
        S, sm_scale,
        H=H, HKV=HKV, D=D, GRP=H // HKV,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=nw, num_stages=ns,
    )

    return F.linear(out, o_proj_weight)
