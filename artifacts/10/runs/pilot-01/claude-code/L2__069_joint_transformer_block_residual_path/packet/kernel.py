import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem constants (fixed by definition.json)
# ---------------------------------------------------------------------------
DIM = 1536
CDIM = 1152
NHEADS = 24
HEAD_DIM = 64
FF = 6144

# constexpr mirrors for use inside @triton.jit kernels
_NHEADS = tl.constexpr(24)
_HEAD_DIM = tl.constexpr(64)


# ---------------------------------------------------------------------------
# Kernel 1: LayerNorm(x) * (1 + scale) + shift  ->  fp16
# ---------------------------------------------------------------------------
@triton.jit
def _ln_mod(
    X, OUT, SCALE, SHIFT,
    rows_per_batch,
    stride_mod,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-6)

    sc = tl.load(SCALE + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(SHIFT + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * (1.0 + sc) + sh
    tl.store(OUT + row * D + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: split fused QKV projection into q/k/v laid out [B, H, T, hd]
# ---------------------------------------------------------------------------
@triton.jit
def _qkv_scatter(
    QKV, BIAS, OUT,
    n_tok,            # tokens per batch for this stream
    tok_off,          # offset of this stream inside T
    T,                # total joint sequence length
    B,                # batch size
    D: tl.constexpr,  # 1536
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)      # row index (b * n_tok + t)
    which = tl.program_id(1)    # 0=q 1=k 2=v
    b = pid // n_tok
    t = pid - b * n_tok

    cols = tl.arange(0, BLOCK)
    mask = cols < D

    src = QKV + pid * (3 * D) + which * D + cols
    x = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    bi = tl.load(BIAS + which * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x + bi).to(tl.float16)

    h = cols // _HEAD_DIM
    d = cols % _HEAD_DIM
    # OUT is [3, B, NHEADS, T, HEAD_DIM]
    dst = (which * B + b) * (_NHEADS * T * _HEAD_DIM) \
        + h * (T * _HEAD_DIM) + (t + tok_off) * _HEAD_DIM + d
    tl.store(OUT + dst, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: attention output [B,H,T,hd] -> two contiguous [rows, DIM] buffers
# ---------------------------------------------------------------------------
@triton.jit
def _attn_gather(
    A, IMG, CTX,
    S, C, T,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid - b * T

    cols = tl.arange(0, BLOCK)
    mask = cols < D
    h = cols // _HEAD_DIM
    d = cols % _HEAD_DIM

    src = A + b * (_NHEADS * T * _HEAD_DIM) + h * (T * _HEAD_DIM) + t * _HEAD_DIM + d
    x = tl.load(src, mask=mask, other=0.0)

    if t < S:
        tl.store(IMG + (b * S + t) * D + cols, x, mask=mask)
    else:
        tl.store(CTX + (b * C + (t - S)) * D + cols, x, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 4: gated residual + LayerNorm + mlp modulation (fused)
#   hs    = res + gate * (proj + bias)                       (fp32, kept)
#   norm2 = LN(hs) * (1 + scale_mlp) + shift_mlp             (fp16)
# ---------------------------------------------------------------------------
@triton.jit
def _res_ln_mod(
    RES, PROJ, PBIAS, GATE, SCALE, SHIFT, HS, OUT,
    rows_per_batch, stride_mod,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    p = tl.load(PROJ + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    pb = tl.load(PBIAS + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    hs = r + g * (p + pb)
    tl.store(HS + row * D + cols, hs, mask=mask)

    mean = tl.sum(hs, axis=0) / D
    xc = tl.where(mask, hs - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-6)

    sc = tl.load(SCALE + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(SHIFT + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * (1.0 + sc) + sh
    tl.store(OUT + row * D + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 5: bias + GELU(tanh) -> fp16
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_bias(X, BIAS, OUT, n_elem, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(BIAS + (offs % D), mask=mask, other=0.0).to(tl.float32)
    x = x + b
    # gelu_tanh(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3))
    u = 1.5957691216057308 * (x + 0.044715 * x * x * x)
    y = x / (1.0 + tl.exp(-u))
    tl.store(OUT + offs, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 6: final gated residual -> fp32
# ---------------------------------------------------------------------------
@triton.jit
def _gated_add(
    HS, FFO, FBIAS, GATE, OUT,
    rows_per_batch, stride_mod,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // rows_per_batch
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    hs = tl.load(HS + row * D + cols, mask=mask, other=0.0)
    f = tl.load(FFO + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    fb = tl.load(FBIAS + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + b * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * D + cols, hs + g * (f + fb), mask=mask)


# ---------------------------------------------------------------------------
def _nw(d):
    return 8 if d >= 1024 else 4


def _ln_mod_launch(x, scale, shift, rows_per_batch, D):
    R = x.shape[0]
    out = torch.empty((R, D), dtype=torch.float16, device=x.device)
    _ln_mod[(R,)](
        x, out, scale, shift, rows_per_batch, scale.stride(0),
        D=D, BLOCK=triton.next_power_of_2(D), num_warps=_nw(D),
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm1_context_weight: torch.Tensor,
    norm1_context_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    add_q_proj_weight: torch.Tensor,
    add_q_proj_bias: torch.Tensor,
    add_k_proj_weight: torch.Tensor,
    add_k_proj_bias: torch.Tensor,
    add_v_proj_weight: torch.Tensor,
    add_v_proj_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    to_add_out_weight: torch.Tensor,
    to_add_out_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    ff_context_linear1_weight: torch.Tensor,
    ff_context_linear1_bias: torch.Tensor,
    ff_context_linear2_weight: torch.Tensor,
    ff_context_linear2_bias: torch.Tensor,
):
    dev = hidden_states.device
    B, S, _ = hidden_states.shape
    C = encoder_hidden_states.shape[1]
    T = S + C
    N = B * S
    M = B * C

    hs_in = hidden_states.contiguous()
    eh_in = encoder_hidden_states.contiguous()

    # ---- modulation parameters (fp32; weights are read once, memory bound) ----
    ts = F.silu(temb)
    mod = torch.addmm(norm1_bias, ts, norm1_weight.t())
    modc = torch.addmm(norm1_context_bias, ts, norm1_context_weight.t())
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    (c_shift_msa, c_scale_msa, c_gate_msa,
     c_shift_mlp, c_scale_mlp, c_gate_mlp) = modc.chunk(6, dim=-1)

    # ---- weight prep: fp32 -> fp16 (+ qkv fusion) ----
    wqkv_i = torch.empty((3 * DIM, DIM), dtype=torch.float16, device=dev)
    wqkv_i[0:DIM].copy_(to_q_weight)
    wqkv_i[DIM:2 * DIM].copy_(to_k_weight)
    wqkv_i[2 * DIM:3 * DIM].copy_(to_v_weight)
    bqkv_i = torch.cat([to_q_bias, to_k_bias, to_v_bias])

    wqkv_c = torch.empty((3 * DIM, CDIM), dtype=torch.float16, device=dev)
    wqkv_c[0:DIM].copy_(add_q_proj_weight)
    wqkv_c[DIM:2 * DIM].copy_(add_k_proj_weight)
    wqkv_c[2 * DIM:3 * DIM].copy_(add_v_proj_weight)
    bqkv_c = torch.cat([add_q_proj_bias, add_k_proj_bias, add_v_proj_bias])

    w_out = to_out_weight.half()
    w_addout = to_add_out_weight.half()
    w_ff1 = ff_linear1_weight.half()
    w_ff2 = ff_linear2_weight.half()
    w_cff1 = ff_context_linear1_weight.half()
    w_cff2 = ff_context_linear2_weight.half()

    # ---- norm + msa modulation ----
    nh = _ln_mod_launch(hs_in.view(N, DIM), scale_msa, shift_msa, S, DIM)
    ne = _ln_mod_launch(eh_in.view(M, CDIM), c_scale_msa, c_shift_msa, C, CDIM)

    # ---- fused QKV projections ----
    qkv_i = torch.mm(nh, wqkv_i.t())          # [N, 3*DIM]
    qkv_c = torch.mm(ne, wqkv_c.t())          # [M, 3*DIM]

    qkv = torch.empty((3, B, NHEADS, T, HEAD_DIM), dtype=torch.float16, device=dev)

    _qkv_scatter[(N, 3)](qkv_i, bqkv_i, qkv, S, 0, T, B,
                         D=DIM, BLOCK=2048, num_warps=8)
    _qkv_scatter[(M, 3)](qkv_c, bqkv_c, qkv, C, S, T, B,
                         D=DIM, BLOCK=2048, num_warps=8)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # ---- joint attention ----
    a = F.scaled_dot_product_attention(q, k, v)

    attn_i = torch.empty((N, DIM), dtype=torch.float16, device=dev)
    attn_c = torch.empty((M, DIM), dtype=torch.float16, device=dev)
    _attn_gather[(B * T,)](a, attn_i, attn_c, S, C, T,
                           D=DIM, BLOCK=2048, num_warps=8)

    # ---- output projections ----
    proj_i = torch.mm(attn_i, w_out.t())      # [N, DIM]
    proj_c = torch.mm(attn_c, w_addout.t())   # [M, CDIM]

    # ---- gated residual + norm + mlp modulation ----
    hs = torch.empty((N, DIM), dtype=torch.float32, device=dev)
    nh2 = torch.empty((N, DIM), dtype=torch.float16, device=dev)
    _res_ln_mod[(N,)](hs_in.view(N, DIM), proj_i, to_out_bias, gate_msa,
                      scale_mlp, shift_mlp, hs, nh2, S, mod.stride(0),
                      D=DIM, BLOCK=2048, num_warps=8)

    eh = torch.empty((M, CDIM), dtype=torch.float32, device=dev)
    ne2 = torch.empty((M, CDIM), dtype=torch.float16, device=dev)
    _res_ln_mod[(M,)](eh_in.view(M, CDIM), proj_c, to_add_out_bias, c_gate_msa,
                      c_scale_mlp, c_shift_mlp, eh, ne2, C, modc.stride(0),
                      D=CDIM, BLOCK=triton.next_power_of_2(CDIM), num_warps=8)

    # ---- feed forward ----
    h1 = torch.mm(nh2, w_ff1.t())             # [N, FF]
    g1 = torch.empty_like(h1)
    n1 = N * FF
    _gelu_bias[(triton.cdiv(n1, 2048),)](h1, ff_linear1_bias, g1, n1,
                                         D=FF, BLOCK=2048, num_warps=8)
    ffo = torch.mm(g1, w_ff2.t())             # [N, DIM]

    h2 = torch.mm(ne2, w_cff1.t())            # [M, FF]
    g2 = torch.empty_like(h2)
    n2 = M * FF
    _gelu_bias[(triton.cdiv(n2, 2048),)](h2, ff_context_linear1_bias, g2, n2,
                                         D=FF, BLOCK=2048, num_warps=8)
    cffo = torch.mm(g2, w_cff2.t())           # [M, CDIM]

    # ---- final gated residual ----
    out_h = torch.empty((N, DIM), dtype=torch.float32, device=dev)
    _gated_add[(N,)](hs, ffo, ff_linear2_bias, gate_mlp, out_h, S, mod.stride(0),
                     D=DIM, BLOCK=2048, num_warps=8)

    out_e = torch.empty((M, CDIM), dtype=torch.float32, device=dev)
    _gated_add[(M,)](eh, cffo, ff_context_linear2_bias, c_gate_mlp, out_e, C,
                     modc.stride(0), D=CDIM,
                     BLOCK=triton.next_power_of_2(CDIM), num_warps=8)

    return out_e.view(B, C, CDIM), out_h.view(B, S, DIM)
