import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused softmax-backward:
#   d_i   = sum_j  fp32(aw[i,j]) * fp32(gaw[i,j])
#   gas   = bf16( fp32(bf16( aw * (gaw - d) )) * scaling )
# One pass over the row; matches the reference's intermediate bf16 rounding.
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_bwd(AW, GAW, GAS, S, scaling, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * S
    cols = tl.arange(0, BLOCK)
    mask = cols < S
    aw = tl.load(AW + base + cols, mask=mask, other=0.0).to(tl.float32)
    gaw = tl.load(GAW + base + cols, mask=mask, other=0.0).to(tl.float32)
    d = tl.sum(aw * gaw, axis=0)
    x = (aw * (gaw - d)).to(tl.bfloat16).to(tl.float32) * scaling
    tl.store(GAS + base + cols, x.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# RoPE backward for Q  +  [B,H,S,D] -> [B*S, H*D] transpose, writing straight
# into the concatenated  [gq | gk | gv]  buffer.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_q(DQ, COS, SIN, OUT, S, OUT_STRIDE,
                NH: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // S).to(tl.int64)
    s = (pid % S).to(tl.int64)
    h = tl.arange(0, NH)[:, None].to(tl.int64)
    d = tl.arange(0, HD)[None, :]
    dr = (d + HALF) % HD

    src = DQ + b * (NH * S * HD) + h * (S * HD) + s * HD
    a = tl.load(src + d).to(tl.float32)
    ar = tl.load(src + dr).to(tl.float32)

    cb = b * (S * HD) + s * HD
    c = tl.load(COS + cb + d).to(tl.float32)
    sn = tl.load(SIN + cb + dr).to(tl.float32)

    p = (a * c).to(tl.bfloat16).to(tl.float32)
    q = (ar * sn).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(d < HALF, 1.0, -1.0)
    out = (p + sign * q).to(tl.bfloat16)
    tl.store(OUT + pid.to(tl.int64) * OUT_STRIDE + h * HD + d, out)


# ---------------------------------------------------------------------------
# GQA group reduce (+ optional RoPE backward) for K and V.
# in : [B, NKV*G, S, HD]   out: [B*S, NKV*HD] slice of the concat buffer
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_kv(DK, COS, SIN, OUT, S, OUT_STRIDE,
                 NKV: tl.constexpr, G: tl.constexpr, HD: tl.constexpr,
                 HALF: tl.constexpr, DO_ROPE: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // S).to(tl.int64)
    s = (pid % S).to(tl.int64)
    kv = tl.arange(0, NKV)[:, None].to(tl.int64)
    d = tl.arange(0, HD)[None, :]

    base = DK + b * (NKV * G * S * HD) + s * HD + kv * (G * S * HD)
    acc = tl.zeros([NKV, HD], dtype=tl.float32)
    for g in tl.static_range(G):
        acc += tl.load(base + g * (S * HD) + d).to(tl.float32)
    gk = acc.to(tl.bfloat16)

    if DO_ROPE:
        dr = (d + HALF) % HD
        accr = tl.zeros([NKV, HD], dtype=tl.float32)
        for g in tl.static_range(G):
            accr += tl.load(base + g * (S * HD) + dr).to(tl.float32)
        gkr = accr.to(tl.bfloat16).to(tl.float32)

        cb = b * (S * HD) + s * HD
        c = tl.load(COS + cb + d).to(tl.float32)
        sn = tl.load(SIN + cb + dr).to(tl.float32)
        p = (gk.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        q = (gkr * sn).to(tl.bfloat16).to(tl.float32)
        sign = tl.where(d < HALF, 1.0, -1.0)
        out = (p + sign * q).to(tl.bfloat16)
    else:
        out = gk

    tl.store(OUT + pid.to(tl.int64) * OUT_STRIDE + kv * HD + d, out)


def _nw(block):
    if block <= 128:
        return 1
    if block <= 512:
        return 2
    if block <= 1024:
        return 4
    return 8


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    scaling: float,
):
    B, S, H = hidden_states.shape
    M = B * S
    NH, NKV, HD = 32, 8, 128
    G = NH // NKV
    QP = NH * HD
    KP = NKV * HD

    go2 = grad_output.reshape(M, H)
    ao2 = attn_output.reshape(M, QP)

    # 1. output projection
    dAO = torch.mm(go2, o_weight)                       # [M, QP]
    grad_o_weight = torch.mm(go2.t(), ao2)              # [H, QP]

    # 2. [B,S,NH,HD] -> [B,NH,S,HD]
    dAO4 = dAO.view(B, S, NH, HD).transpose(1, 2).contiguous()

    # 3. attention backward
    gaw = torch.matmul(dAO4, value_states.transpose(2, 3))       # [B,NH,S,S]
    dV = torch.matmul(attn_weights.transpose(2, 3), dAO4)        # [B,NH,S,HD]

    # 4./5. softmax backward + scaling (fused)
    gas = torch.empty_like(gaw)
    BLOCK = triton.next_power_of_2(S)
    _softmax_bwd[(B * NH * S,)](attn_weights, gaw, gas, S, scaling,
                                BLOCK=BLOCK, num_warps=_nw(BLOCK))

    # 6. grads through Q@K^T
    dQ = torch.matmul(gas, key_states)                           # [B,NH,S,HD]
    dK = torch.matmul(gas.transpose(2, 3), query_states)         # [B,NH,S,HD]

    # 7./8./9. group reduce + RoPE backward + transpose, straight into
    #          the concatenated [gq | gk | gv] buffer.
    cat_g = torch.empty((M, QP + 2 * KP), dtype=torch.bfloat16, device=dAO.device)
    _rope_bwd_q[(M,)](dQ, cos, sin, cat_g, S, cat_g.stride(0),
                      NH=NH, HD=HD, HALF=HD // 2, num_warps=4)
    _rope_bwd_kv[(M,)](dK, cos, sin, cat_g[:, QP:], S, cat_g.stride(0),
                       NKV=NKV, G=G, HD=HD, HALF=HD // 2, DO_ROPE=True,
                       num_warps=2)
    _rope_bwd_kv[(M,)](dV, cos, sin, cat_g[:, QP + KP:], S, cat_g.stride(0),
                       NKV=NKV, G=G, HD=HD, HALF=HD // 2, DO_ROPE=False,
                       num_warps=2)

    # 10. Q/K/V projections
    cat_w = torch.cat((q_weight, k_weight, v_weight), dim=0)     # [QP+2KP, H]
    grad_hidden = torch.mm(cat_g, cat_w).view(B, S, H)
    gw = torch.mm(cat_g.t(), hidden_states.reshape(M, H))        # [QP+2KP, H]

    return (
        grad_hidden,
        gw[:QP],
        gw[QP:QP + KP],
        gw[QP + KP:],
        grad_o_weight,
    )
