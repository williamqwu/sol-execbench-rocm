import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NH = 28
HD = 128
HALF = 64
H = NH * HD  # 3584


# ---------------------------------------------------------------------------
# Fused: read qkv linear output [S, 3H], apply 2D RoPE to q and k, and scatter
# into padded per-sequence buffers [n, NH, Lmax, HD] (zero-filled padding).
# ---------------------------------------------------------------------------
@triton.jit
def _rope_pad_kernel(
    QKV, COS, SIN, CU, QP, KP, VP,
    Lmax,
    NH_: tl.constexpr, HD_: tl.constexpr, BL: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.program_id(2)

    start = tl.load(CU + b).to(tl.int32)
    end = tl.load(CU + b + 1).to(tl.int32)
    L = end - start

    p = pb * BL + tl.arange(0, BL)
    d = tl.arange(0, HD_)
    # rotate_half(x)[d] = -x[d+64] (d<64) ; x[d-64] (d>=64)
    d_rot = (d + (HD_ // 2)) % HD_
    sgn = tl.where(d < (HD_ // 2), -1.0, 1.0)

    inb = p < L
    s = start + p

    row3 = s[:, None] * (3 * NH_ * HD_)
    qoff = row3 + h * HD_ + d[None, :]
    qoff_r = row3 + h * HD_ + d_rot[None, :]
    koff = row3 + NH_ * HD_ + h * HD_ + d[None, :]
    koff_r = row3 + NH_ * HD_ + h * HD_ + d_rot[None, :]
    voff = row3 + 2 * NH_ * HD_ + h * HD_ + d[None, :]

    m = inb[:, None]
    q = tl.load(QKV + qoff, mask=m, other=0.0)
    qr = tl.load(QKV + qoff_r, mask=m, other=0.0)
    k = tl.load(QKV + koff, mask=m, other=0.0)
    kr = tl.load(QKV + koff_r, mask=m, other=0.0)
    v = tl.load(QKV + voff, mask=m, other=0.0)

    cs = tl.load(COS + s[:, None] * HD_ + d[None, :], mask=m, other=0.0)
    sn = tl.load(SIN + s[:, None] * HD_ + d[None, :], mask=m, other=0.0)

    q_rope = q * cs + (sgn[None, :] * qr) * sn
    k_rope = k * cs + (sgn[None, :] * kr) * sn

    dst = ((b * NH_ + h) * Lmax + p[:, None]) * HD_ + d[None, :]
    sm = (p < Lmax)[:, None]
    tl.store(QP + dst, tl.where(m, q_rope, 0.0), mask=sm)
    tl.store(KP + dst, tl.where(m, k_rope, 0.0), mask=sm)
    tl.store(VP + dst, tl.where(m, v, 0.0), mask=sm)


# ---------------------------------------------------------------------------
# Dense [S, NH*HD] -> padded [n, NH, Lmax, HD]
# ---------------------------------------------------------------------------
@triton.jit
def _pad_kernel(SRC, CU, DST, Lmax, NH_: tl.constexpr, HD_: tl.constexpr, BL: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.program_id(2)
    start = tl.load(CU + b).to(tl.int32)
    end = tl.load(CU + b + 1).to(tl.int32)
    L = end - start
    p = pb * BL + tl.arange(0, BL)
    d = tl.arange(0, HD_)
    inb = p < L
    s = start + p
    x = tl.load(SRC + s[:, None] * (NH_ * HD_) + h * HD_ + d[None, :], mask=inb[:, None], other=0.0)
    dst = ((b * NH_ + h) * Lmax + p[:, None]) * HD_ + d[None, :]
    tl.store(DST + dst, tl.where(inb[:, None], x, 0.0), mask=(p < Lmax)[:, None])


# ---------------------------------------------------------------------------
# Padded [n, NH, Lmax, HD] -> dense [S, NH*HD]
# ---------------------------------------------------------------------------
@triton.jit
def _unpad_kernel(SRC, CU, DST, Lmax, NH_: tl.constexpr, HD_: tl.constexpr, BL: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.program_id(2)
    start = tl.load(CU + b).to(tl.int32)
    end = tl.load(CU + b + 1).to(tl.int32)
    L = end - start
    p = pb * BL + tl.arange(0, BL)
    d = tl.arange(0, HD_)
    inb = p < L
    src = ((b * NH_ + h) * Lmax + p[:, None]) * HD_ + d[None, :]
    x = tl.load(SRC + src, mask=inb[:, None], other=0.0)
    s = start + p
    tl.store(DST + s[:, None] * (NH_ * HD_) + h * HD_ + d[None, :], x, mask=inb[:, None])


# ---------------------------------------------------------------------------
# scores*scaling -> mask -> softmax -> zero padded query rows.  Fused, one pass.
# ---------------------------------------------------------------------------
@triton.jit
def _masked_softmax_kernel(
    SC, CU, OUT, Lmax, scaling,
    NHL,  # NH * Lmax
    ROWS_PER: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in tl.static_range(ROWS_PER):
        r = pid * ROWS_PER + i
        b = r // NHL
        p = (r - b * NHL) % Lmax
        start = tl.load(CU + b).to(tl.int32)
        end = tl.load(CU + b + 1).to(tl.int32)
        L = end - start
        cols = tl.arange(0, BN)
        x = tl.load(SC + r.to(tl.int64) * Lmax + cols, mask=cols < Lmax, other=0.0)
        x = x * scaling
        x = tl.where(cols < L, x, float("-inf"))
        mx = tl.max(x, 0)
        e = tl.exp(x - mx)
        ssum = tl.sum(e, 0)
        y = e / ssum
        y = tl.where(p < L, y, 0.0)
        tl.store(OUT + r.to(tl.int64) * Lmax + cols, y, mask=cols < Lmax)


# ---------------------------------------------------------------------------
# softmax backward: gs = w * (gw - sum(gw*w)) * scaling
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_bwd_kernel(GW, W, OUT, Lmax, scaling, ROWS_PER: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.static_range(ROWS_PER):
        r = (pid * ROWS_PER + i).to(tl.int64)
        cols = tl.arange(0, BN)
        m = cols < Lmax
        base = r * Lmax + cols
        gw = tl.load(GW + base, mask=m, other=0.0)
        w = tl.load(W + base, mask=m, other=0.0)
        sg = tl.sum(gw * w, 0)
        o = w * (gw - sg)
        o = o * scaling
        tl.store(OUT + base, o, mask=m)


# ---------------------------------------------------------------------------
# Inverse RoPE on grad_q / grad_k, then scatter q,k,v grads into grad_qkv[S,3H]
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_scatter_kernel(
    GQ, GK, GV, COS, SIN, CU, GQKV,
    Lmax,
    NH_: tl.constexpr, HD_: tl.constexpr, BL: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.program_id(2)
    start = tl.load(CU + b).to(tl.int32)
    end = tl.load(CU + b + 1).to(tl.int32)
    L = end - start
    p = pb * BL + tl.arange(0, BL)
    d = tl.arange(0, HD_)
    d_rot = (d + (HD_ // 2)) % HD_
    # rotate_half_inverse(x)[d] = x[d+64] (d<64) ; -x[d-64] (d>=64)
    sgn = tl.where(d < (HD_ // 2), 1.0, -1.0)

    inb = p < L
    m = inb[:, None]
    s = start + p

    src = ((b * NH_ + h) * Lmax + p[:, None]) * HD_
    src_d = src + d[None, :]
    src_r = src + d_rot[None, :]

    cs = tl.load(COS + s[:, None] * HD_ + d[None, :], mask=m, other=0.0)
    sn = tl.load(SIN + s[:, None] * HD_ + d[None, :], mask=m, other=0.0)
    sn_r = tl.load(SIN + s[:, None] * HD_ + d_rot[None, :], mask=m, other=0.0)

    gq = tl.load(GQ + src_d, mask=m, other=0.0)
    gq_r = tl.load(GQ + src_r, mask=m, other=0.0)
    gk = tl.load(GK + src_d, mask=m, other=0.0)
    gk_r = tl.load(GK + src_r, mask=m, other=0.0)
    gv = tl.load(GV + src_d, mask=m, other=0.0)

    oq = gq * cs + sgn[None, :] * (gq_r * sn_r)
    ok = gk * cs + sgn[None, :] * (gk_r * sn_r)

    row3 = s[:, None] * (3 * NH_ * HD_) + h * HD_ + d[None, :]
    tl.store(GQKV + row3, oq, mask=m)
    tl.store(GQKV + row3 + NH_ * HD_, ok, mask=m)
    tl.store(GQKV + row3 + 2 * NH_ * HD_, gv, mask=m)


def _grid_bl(Lmax):
    if Lmax <= 16:
        return 16
    if Lmax <= 32:
        return 32
    return 64


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_dropout: float,
    scaling: float,
):
    S = hidden_states.shape[0]
    dev = hidden_states.device
    f32 = torch.float32

    # ---- QKV projection (forward recompute) ----
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)  # [S, 3H]

    cu = cu_seqlens.contiguous()
    lens = (cu[1:] - cu[:-1]).tolist()
    n = len(lens)
    Lmax = max(lens) if n > 0 else 0

    nb = n * NH
    BL = _grid_bl(Lmax)
    grid = (n, NH, triton.cdiv(Lmax, BL))

    qp = torch.empty((nb, Lmax, HD), device=dev, dtype=f32)
    kp = torch.empty((nb, Lmax, HD), device=dev, dtype=f32)
    vp = torch.empty((nb, Lmax, HD), device=dev, dtype=f32)

    _rope_pad_kernel[grid](
        qkv, cos, sin, cu, qp, kp, vp, Lmax,
        NH_=NH, HD_=HD, BL=BL, num_warps=4,
    )

    # ---- attention forward ----
    scores = torch.bmm(qp, kp.transpose(1, 2))  # [nb, Lmax, Lmax]
    BN = max(16, triton.next_power_of_2(Lmax))
    nrows = nb * Lmax
    rows_per = 1
    w = torch.empty_like(scores)
    _masked_softmax_kernel[(nrows // rows_per,)](
        scores, cu, w, Lmax, scaling, NH * Lmax,
        ROWS_PER=rows_per, BN=BN, num_warps=4,
    )
    ao = torch.bmm(w, vp)  # [nb, Lmax, HD]

    attn_out = torch.empty((S, H), device=dev, dtype=f32)
    _unpad_kernel[grid](ao, cu, attn_out, Lmax, NH_=NH, HD_=HD, BL=BL, num_warps=4)

    # ---- output projection backward ----
    grad_proj_bias = grad_output.sum(dim=0)
    grad_proj_weight = grad_output.t() @ attn_out
    grad_attn_output = grad_output @ proj_weight  # [S, H]

    gp = torch.empty((nb, Lmax, HD), device=dev, dtype=f32)
    _pad_kernel[grid](grad_attn_output, cu, gp, Lmax, NH_=NH, HD_=HD, BL=BL, num_warps=4)

    # ---- attention backward ----
    gw = torch.bmm(gp, vp.transpose(1, 2))          # [nb, Lmax, Lmax]
    gv = torch.bmm(w.transpose(1, 2), gp)           # [nb, Lmax, HD]
    _softmax_bwd_kernel[(nrows // rows_per,)](
        gw, w, gw, Lmax, scaling, ROWS_PER=rows_per, BN=BN, num_warps=4,
    )
    gq = torch.bmm(gw, kp)                          # [nb, Lmax, HD]
    gk = torch.bmm(gw.transpose(1, 2), qp)          # [nb, Lmax, HD]

    # ---- inverse RoPE + scatter ----
    grad_qkv = torch.empty((S, 3 * H), device=dev, dtype=f32)
    _rope_bwd_scatter_kernel[grid](
        gq, gk, gv, cos, sin, cu, grad_qkv, Lmax,
        NH_=NH, HD_=HD, BL=BL, num_warps=4,
    )

    # ---- QKV projection backward ----
    grad_hidden_states = grad_qkv @ qkv_weight
    grad_qkv_weight = grad_qkv.t() @ hidden_states
    grad_qkv_bias = grad_qkv.sum(dim=0)

    return grad_hidden_states, grad_qkv_weight, grad_qkv_bias, grad_proj_weight, grad_proj_bias
