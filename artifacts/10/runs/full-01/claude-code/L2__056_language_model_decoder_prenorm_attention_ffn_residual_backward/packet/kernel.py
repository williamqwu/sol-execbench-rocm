import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# SwiGLU backward:  writes grad_gate and grad_up into one fused buffer
#   G[:, :I]  = grad_gate = grad_swiglu * silu_up                     (bf16)
#   G[:, I:]  = grad_up   = (grad_swiglu * gate) * bf16(grad_silu)    (bf16)
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_bwd_kernel(
    GSW, GATE, UP, SILU_UP, OUT,
    M, I,
    stride_out_m,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    base = row.to(tl.int64) * I + offs

    gsw = tl.load(GSW + base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + base, mask=mask, other=0.0).to(tl.float32)
    silu = tl.load(SILU_UP + base, mask=mask, other=0.0).to(tl.float32)

    # grad_gate = grad_swiglu_output * silu_up   (bf16 * bf16 -> bf16)
    g_gate = (gsw * silu).to(tl.bfloat16)

    sig = tl.sigmoid(up)
    grad_silu = sig * (1.0 + up * (1.0 - sig))
    grad_silu_bf = grad_silu.to(tl.bfloat16)

    # grad_up = (grad_swiglu_output * gate) * grad_silu.to(bf16)
    t = (gsw * gate).to(tl.bfloat16)
    g_up = (t.to(tl.float32) * grad_silu_bf.to(tl.float32)).to(tl.bfloat16)

    obase = row.to(tl.int64) * stride_out_m + offs
    tl.store(OUT + obase, g_gate, mask=mask)
    tl.store(OUT + obase + I, g_up, mask=mask)


def swiglu_bwd(gsw, gate, up, silu_up, M, I):
    out = torch.empty((M, 2 * I), dtype=torch.bfloat16, device=gsw.device)
    BLOCK = 1024
    grid = (triton.cdiv(I, BLOCK), M)
    _swiglu_bwd_kernel[grid](
        gsw, gate, up, silu_up, out, M, I, out.stride(0),
        BLOCK=BLOCK, num_warps=4, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Softmax backward (rowwise) + scale by 1/sqrt(head_dim)
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_bwd_kernel(
    GAW, AW, OUT, S, sqrt_hd,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < S
    base = row * S + offs
    g = tl.load(GAW + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(AW + base, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(g * w, axis=0)
    o = w * (g - s)
    o = o / sqrt_hd
    tl.store(OUT + base, o.to(tl.bfloat16), mask=mask)


def softmax_bwd(gaw, aw, BH, S, sqrt_hd):
    out = torch.empty_like(gaw)
    BLOCK = triton.next_power_of_2(S)
    nw = 4 if BLOCK <= 512 else 8
    _softmax_bwd_kernel[(BH * S,)](
        gaw, aw, out, S, sqrt_hd, BLOCK=BLOCK, num_warps=nw, num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# RoPE backward for Q:  (B, H, S, D) -> (B, S, H*D) slice of a fused buffer
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_q_kernel(
    X, COS, SIN, OUT,
    B, H, S, D: tl.constexpr, HD: tl.constexpr,
    stride_out_m, out_off,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H
    offs_s = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HD
    m = mask_s[:, None] & mask_d[None, :]

    xb = (pid_bh.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * D + offs_d[None, :]
    x1 = tl.load(X + xb, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X + xb + HD, mask=m, other=0.0).to(tl.float32)

    cb = (b.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * D + offs_d[None, :]
    c1 = tl.load(COS + cb, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + HD, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + HD, mask=m, other=0.0).to(tl.float32)

    a1 = (x1 * c1).to(tl.bfloat16).to(tl.float32)
    b1 = (x2 * s1).to(tl.bfloat16).to(tl.float32)
    o1 = (a1 + b1).to(tl.bfloat16)

    a2 = (x2 * c2).to(tl.bfloat16).to(tl.float32)
    b2 = (-(x1 * s2)).to(tl.bfloat16).to(tl.float32)
    o2 = (a2 + b2).to(tl.bfloat16)

    ob = (b.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * stride_out_m \
        + out_off + h.to(tl.int64) * D + offs_d[None, :]
    tl.store(OUT + ob, o1, mask=m)
    tl.store(OUT + ob + HD, o2, mask=m)


# ---------------------------------------------------------------------------
# GQA group-reduce + RoPE backward for K
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_k_kernel(
    X, COS, SIN, OUT,
    B, HKV: tl.constexpr, G: tl.constexpr, S, D: tl.constexpr, HD: tl.constexpr,
    stride_out_m, out_off,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // HKV
    hk = pid_bh % HKV
    offs_s = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HD
    m = mask_s[:, None] & mask_d[None, :]

    x1 = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    x2 = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for g in tl.static_range(G):
        head = hk * G + g
        xb = ((b.to(tl.int64) * (HKV * G) + head) * S + offs_s[:, None].to(tl.int64)) * D + offs_d[None, :]
        x1 += tl.load(X + xb, mask=m, other=0.0).to(tl.float32)
        x2 += tl.load(X + xb + HD, mask=m, other=0.0).to(tl.float32)
    # torch sum over bf16 accumulates in fp32 then rounds
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)

    cb = (b.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * D + offs_d[None, :]
    c1 = tl.load(COS + cb, mask=m, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + HD, mask=m, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb, mask=m, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + HD, mask=m, other=0.0).to(tl.float32)

    a1 = (x1 * c1).to(tl.bfloat16).to(tl.float32)
    b1 = (x2 * s1).to(tl.bfloat16).to(tl.float32)
    o1 = (a1 + b1).to(tl.bfloat16)

    a2 = (x2 * c2).to(tl.bfloat16).to(tl.float32)
    b2 = (-(x1 * s2)).to(tl.bfloat16).to(tl.float32)
    o2 = (a2 + b2).to(tl.bfloat16)

    ob = (b.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * stride_out_m \
        + out_off + hk.to(tl.int64) * D + offs_d[None, :]
    tl.store(OUT + ob, o1, mask=m)
    tl.store(OUT + ob + HD, o2, mask=m)


# ---------------------------------------------------------------------------
# GQA group-reduce for V (no RoPE)
# ---------------------------------------------------------------------------
@triton.jit
def _gqa_reduce_v_kernel(
    X, OUT,
    B, HKV: tl.constexpr, G: tl.constexpr, S, D: tl.constexpr,
    stride_out_m, out_off,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // HKV
    hk = pid_bh % HKV
    offs_s = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    m = mask_s[:, None] & mask_d[None, :]

    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for g in tl.static_range(G):
        head = hk * G + g
        xb = ((b.to(tl.int64) * (HKV * G) + head) * S + offs_s[:, None].to(tl.int64)) * D + offs_d[None, :]
        acc += tl.load(X + xb, mask=m, other=0.0).to(tl.float32)

    ob = (b.to(tl.int64) * S + offs_s[:, None].to(tl.int64)) * stride_out_m \
        + out_off + hk.to(tl.int64) * D + offs_d[None, :]
    tl.store(OUT + ob, acc.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------
# RMSNorm backward (input grad + weight grad partials), fused with residual add
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_bwd_kernel(
    G, HSN, W, X, VAR, ADD, OUT, GW_PART,
    M, N, eps, inv_n,
    ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    gw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    start = pid * ROWS
    for i in range(ROWS):
        row = start + i
        if row < M:
            base = row.to(tl.int64) * N + offs
            gf = tl.load(G + base, mask=mask, other=0.0).to(tl.float32)
            hsn = tl.load(HSN + base, mask=mask, other=0.0)
            gw_acc += gf * hsn

            xf = tl.load(X + base, mask=mask, other=0.0).to(tl.float32)
            var = tl.load(VAR + row)
            rs = 1.0 / tl.sqrt(var + eps)

            gn = gf * w
            gvar = -0.5 * tl.sum(gn * xf, axis=0) * rs * rs * rs
            gh = gn * rs + (2.0 * inv_n) * xf * gvar
            ghb = gh.to(tl.bfloat16).to(tl.float32)

            add = tl.load(ADD + base, mask=mask, other=0.0).to(tl.float32)
            tl.store(OUT + base, (ghb + add).to(tl.bfloat16), mask=mask)

    tl.store(GW_PART + pid.to(tl.int64) * N + offs, gw_acc, mask=mask)


def rmsnorm_bwd(g, hsn, w, x, var, add, M, N, eps):
    out = torch.empty((M, N), dtype=torch.bfloat16, device=g.device)
    nprog = min(M, 2048)
    rows = triton.cdiv(M, nprog)
    grid_n = triton.cdiv(M, rows)
    gw_part = torch.empty((grid_n, N), dtype=torch.float32, device=g.device)
    BLOCK_N = triton.next_power_of_2(N)
    _rmsnorm_bwd_kernel[(grid_n,)](
        g, hsn, w, x, var, add, out, gw_part,
        M, N, eps, 1.0 / N,
        ROWS=rows, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1,
    )
    return out, gw_part.sum(dim=0)


# ---------------------------------------------------------------------------


def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states_rotated: torch.Tensor,
    key_states_rotated: torch.Tensor,
    key_states_repeated: torch.Tensor,
    value_states_repeated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    residual2: torch.Tensor,
    ffn_input: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    silu_up: torch.Tensor,
    swiglu_output: torch.Tensor,
    input_ln_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    post_attn_ln_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    variance1: torch.Tensor,
    variance2: torch.Tensor,
    hidden_states_normalized1: torch.Tensor,
    hidden_states_normalized2: torch.Tensor,
    eps: float,
):
    B, S, H = grad_output.shape
    NH = 32
    NKV = 8
    HD = 160
    I = 14336
    G = NH // NKV
    M = B * S
    NHD = NH * HD          # 5120
    NKVD = NKV * HD        # 1280

    go = grad_output.reshape(M, H)

    # ---------------- FFN backward ----------------
    gsw = torch.mm(go, down_weight)                              # (M, I)
    grad_down_weight = torch.mm(go.t(), swiglu_output.reshape(M, I))

    Gbuf = swiglu_bwd(gsw, gate.reshape(M, I), up.reshape(M, I),
                      silu_up.reshape(M, I), M, I)
    del gsw
    g_gate = Gbuf[:, :I]
    g_up = Gbuf[:, I:]

    gw_fused = torch.mm(Gbuf.t(), ffn_input.reshape(M, H))       # (2I, H)
    grad_gate_weight = gw_fused[:I]
    grad_up_weight = gw_fused[I:]

    grad_ffn_input = torch.mm(g_gate, gate_weight)
    grad_ffn_input += torch.mm(g_up, up_weight)
    del Gbuf

    gha, grad_post_attn_ln_weight = rmsnorm_bwd(
        grad_ffn_input, hidden_states_normalized2.reshape(M, H),
        post_attn_ln_weight, residual2.reshape(M, H),
        variance2.reshape(M), go, M, H, eps,
    )
    del grad_ffn_input

    # ---------------- Attention backward ----------------
    grad_attn_out = torch.mm(gha, o_weight)                      # (M, NHD)
    grad_o_weight = torch.mm(gha.t(), attn_output.reshape(M, NHD))

    BH = B * NH
    gao = grad_attn_out.view(B, S, NH, HD).transpose(1, 2).reshape(BH, S, HD)
    del grad_attn_out

    aw3 = attn_weights.reshape(BH, S, S)
    vrep = value_states_repeated.reshape(BH, S, HD)
    krep = key_states_repeated.reshape(BH, S, HD)
    qrot = query_states_rotated.reshape(BH, S, HD)

    gaw = torch.bmm(gao, vrep.transpose(1, 2))                   # (BH, S, S)
    gvrep = torch.bmm(aw3.transpose(1, 2), gao)                  # (BH, S, HD)
    del gao

    gl = softmax_bwd(gaw, aw3, BH, S, math.sqrt(HD))
    del gaw

    gqr = torch.bmm(gl, krep)                                    # (BH, S, HD)
    gkrep = torch.bmm(gl.transpose(1, 2), qrot)                  # (BH, S, HD)
    del gl

    # fused QKV grad buffer: [gq | gk | gv] of width NHD + 2*NKVD
    W_QKV = NHD + 2 * NKVD
    QKV = torch.empty((M, W_QKV), dtype=torch.bfloat16, device=go.device)
    sm = QKV.stride(0)

    BLOCK_S = 16
    BLOCK_D = triton.next_power_of_2(HD // 2)
    _rope_bwd_q_kernel[(B * NH, triton.cdiv(S, BLOCK_S))](
        gqr, cos, sin, QKV, B, NH, S, HD, HD // 2, sm, 0,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=4, num_stages=1,
    )
    _rope_bwd_k_kernel[(B * NKV, triton.cdiv(S, BLOCK_S))](
        gkrep, cos, sin, QKV, B, NKV, G, S, HD, HD // 2, sm, NHD,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=4, num_stages=1,
    )
    _gqa_reduce_v_kernel[(B * NKV, triton.cdiv(S, BLOCK_S))](
        gvrep, QKV, B, NKV, G, S, HD, sm, NHD + NKVD,
        BLOCK_S=BLOCK_S, BLOCK_D=triton.next_power_of_2(HD),
        num_warps=4, num_stages=1,
    )
    del gqr, gkrep, gvrep

    gq = QKV[:, :NHD]
    gk = QKV[:, NHD:NHD + NKVD]
    gv = QKV[:, NHD + NKVD:]

    qkvw = torch.mm(QKV.t(), attn_input.reshape(M, H))            # (W_QKV, H)
    grad_q_weight = qkvw[:NHD]
    grad_k_weight = qkvw[NHD:NHD + NKVD]
    grad_v_weight = qkvw[NHD + NKVD:]

    grad_attn_input = torch.mm(gq, q_weight)
    grad_attn_input += torch.mm(gk, k_weight)
    grad_attn_input += torch.mm(gv, v_weight)
    del QKV

    grad_input, grad_input_ln_weight = rmsnorm_bwd(
        grad_attn_input, hidden_states_normalized1.reshape(M, H),
        input_ln_weight, residual.reshape(M, H),
        variance1.reshape(M), gha, M, H, eps,
    )

    return (
        grad_input.view(B, S, H),
        grad_input_ln_weight,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_post_attn_ln_weight,
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
