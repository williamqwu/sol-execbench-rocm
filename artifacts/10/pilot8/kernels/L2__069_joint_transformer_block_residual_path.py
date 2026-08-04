import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused layernorm + adaLN modulation, fp32 in -> fp16 out
# ---------------------------------------------------------------------------
@triton.jit
def _ln_mod(X, Y, SC, SH,
            stride_x, stride_y, stride_m,
            S, D, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // S
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    sc = tl.load(SC + b * stride_m + cols, mask=mask, other=0.0)
    sh = tl.load(SH + b * stride_m + cols, mask=mask, other=0.0)
    y = xc * rstd * (1.0 + sc) + sh
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# residual (h + gate*(a+bias)) -> store fp32; then layernorm+modulate -> fp16
# ---------------------------------------------------------------------------
@triton.jit
def _res_ln_mod(H, A, BIAS, G, SC, SH, HO, NO,
                stride_h, stride_a, stride_m,
                S, D, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // S
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    h = tl.load(H + row * stride_h + cols, mask=mask, other=0.0)
    a = tl.load(A + row * stride_a + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask, other=0.0)
    g = tl.load(G + b * stride_m + cols, mask=mask, other=0.0)
    h = h + g * (a + bias)
    tl.store(HO + row * stride_h + cols, h, mask=mask)
    mean = tl.sum(h, 0) / D
    xc = tl.where(mask, h - mean, 0.0)
    var = tl.sum(xc * xc, 0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    sc = tl.load(SC + b * stride_m + cols, mask=mask, other=0.0)
    sh = tl.load(SH + b * stride_m + cols, mask=mask, other=0.0)
    y = xc * rstd * (1.0 + sc) + sh
    tl.store(NO + row * stride_h + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _res_out(H, A, BIAS, G, OUT,
             stride_h, stride_a, stride_m,
             S, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // S
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    h = tl.load(H + row * stride_h + cols, mask=mask, other=0.0)
    a = tl.load(A + row * stride_a + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask, other=0.0)
    g = tl.load(G + b * stride_m + cols, mask=mask, other=0.0)
    tl.store(OUT + row * stride_h + cols, h + g * (a + bias), mask=mask)


def _ln_mod_call(x, sc, sh, S):
    B, _, D = x.shape
    y = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    BLOCK = triton.next_power_of_2(D)
    _ln_mod[(B * S,)](x, y, sc, sh, D, D, sc.stride(0), S, D,
                      EPS=1e-6, BLOCK=BLOCK, num_warps=8)
    return y


def _res_ln_mod_call(h, a, bias, g, sc, sh, S):
    B, _, D = h.shape
    ho = torch.empty_like(h)
    no = torch.empty(h.shape, dtype=torch.float16, device=h.device)
    BLOCK = triton.next_power_of_2(D)
    _res_ln_mod[(B * S,)](h, a, bias, g, sc, sh, ho, no,
                          D, a.stride(-2), sc.stride(0), S, D,
                          EPS=1e-6, BLOCK=BLOCK, num_warps=8)
    return ho, no


def _res_out_call(h, a, bias, g, S):
    B, _, D = h.shape
    out = torch.empty_like(h)
    BLOCK = triton.next_power_of_2(D)
    _res_out[(B * S,)](h, a, bias, g, out, D, a.stride(-2), g.stride(0), S, D,
                       BLOCK=BLOCK, num_warps=8)
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
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]
    C = encoder_hidden_states.shape[1]
    T = S + C
    dim = 1536
    cdim = 1152
    H = 24
    hd = 64
    dev = hidden_states.device

    # ---- modulation params (fp32, small M) ----
    temb_silu = F.silu(temb)
    mod = F.linear(temb_silu, norm1_weight, norm1_bias)
    modc = F.linear(temb_silu, norm1_context_weight, norm1_context_bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, -1)
    (c_shift_msa, c_scale_msa, c_gate_msa,
     c_shift_mlp, c_scale_mlp, c_gate_mlp) = modc.chunk(6, -1)

    # ---- one flat fp16 buffer for every weight/bias used by a gemm ----
    n_qkv = 3 * dim * dim
    n_add = 3 * dim * cdim
    sizes = [n_qkv, 3 * dim, n_add, 3 * dim,
             dim * dim, cdim * dim,
             6144 * dim, 6144, dim * 6144,
             6144 * cdim, 6144, cdim * 6144]
    total = sum(sizes)
    flat = torch.empty(total, dtype=torch.float16, device=dev)
    off = 0
    views = []
    for s in sizes:
        views.append(flat[off:off + s])
        off += s
    qkvw = views[0].view(3 * dim, dim)
    qkvb = views[1]
    addw = views[2].view(3 * dim, cdim)
    addb = views[3]
    outw = views[4].view(dim, dim)
    aoutw = views[5].view(cdim, dim)
    ff1w = views[6].view(6144, dim)
    ff1b = views[7]
    ff2w = views[8].view(dim, 6144)
    ffc1w = views[9].view(6144, cdim)
    ffc1b = views[10]
    ffc2w = views[11].view(cdim, 6144)

    dsts = [qkvw[0:dim], qkvw[dim:2 * dim], qkvw[2 * dim:],
            qkvb[0:dim], qkvb[dim:2 * dim], qkvb[2 * dim:],
            addw[0:dim], addw[dim:2 * dim], addw[2 * dim:],
            addb[0:dim], addb[dim:2 * dim], addb[2 * dim:],
            outw, aoutw, ff1w, ff1b, ff2w, ffc1w, ffc1b, ffc2w]
    srcs = [to_q_weight, to_k_weight, to_v_weight,
            to_q_bias, to_k_bias, to_v_bias,
            add_q_proj_weight, add_k_proj_weight, add_v_proj_weight,
            add_q_proj_bias, add_k_proj_bias, add_v_proj_bias,
            to_out_weight, to_add_out_weight,
            ff_linear1_weight, ff_linear1_bias, ff_linear2_weight,
            ff_context_linear1_weight, ff_context_linear1_bias,
            ff_context_linear2_weight]
    torch._foreach_copy_(dsts, srcs)

    # ---- pre-attention norm + modulation ----
    nh = _ln_mod_call(hidden_states, scale_msa, shift_msa, S)
    ne = _ln_mod_call(encoder_hidden_states, c_scale_msa, c_shift_msa, C)

    # ---- QKV ----
    qkv_i = torch.addmm(qkvb, nh.view(B * S, dim), qkvw.t()).view(B, S, 3 * dim)
    qkv_c = torch.addmm(addb, ne.view(B * C, cdim), addw.t()).view(B, C, 3 * dim)
    joint = torch.cat([qkv_i, qkv_c], dim=1).view(B, T, 3, H, hd)
    q = joint[:, :, 0].transpose(1, 2)
    k = joint[:, :, 1].transpose(1, 2)
    v = joint[:, :, 2].transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
    attn = attn.transpose(1, 2).reshape(B, T, dim)

    ai = torch.matmul(attn[:, :S], outw.t())
    ac = torch.matmul(attn[:, S:], aoutw.t())

    hs, nh = _res_ln_mod_call(hidden_states, ai, to_out_bias, gate_msa,
                              scale_mlp, shift_mlp, S)
    es, ne = _res_ln_mod_call(encoder_hidden_states, ac, to_add_out_bias,
                              c_gate_msa, c_scale_mlp, c_shift_mlp, C)

    # ---- feed-forward ----
    ffh = F.gelu(torch.addmm(ff1b, nh.view(B * S, dim), ff1w.t()),
                 approximate='tanh')
    ff = torch.mm(ffh, ff2w.t()).view(B, S, dim)
    ffhc = F.gelu(torch.addmm(ffc1b, ne.view(B * C, cdim), ffc1w.t()),
                  approximate='tanh')
    ffc = torch.mm(ffhc, ffc2w.t()).view(B, C, cdim)

    out_h = _res_out_call(hs, ff, ff_linear2_bias, gate_mlp, S)
    out_e = _res_out_call(es, ffc, ff_context_linear2_bias, c_gate_mlp, C)
    return out_e, out_h
