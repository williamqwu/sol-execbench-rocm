import torch
import triton
import triton.language as tl

H = 1280
BLOCK_H = 2048  # >= H, power of two


@triton.jit
def _ln_kernel(X, W, B, OUT, M, eps, H_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H_
    x = tl.load(X + row * H_ + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / H_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / H_
    n = xc / tl.sqrt(var + eps)
    nb = n.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    t = (nb * w).to(tl.bfloat16).to(tl.float32)
    o = (t + b).to(tl.bfloat16)
    tl.store(OUT + row * H_ + cols, o, mask=mask)


@triton.jit
def _gated_res_ln_kernel(RES, A, GATE, W, B, HOUT, OUT, M, eps,
                         H_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H_
    g = tl.load(GATE).to(tl.float32)
    g = tl.extra.libdevice.tanh(g).to(tl.bfloat16).to(tl.float32)
    r = tl.load(RES + row * H_ + cols, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A + row * H_ + cols, mask=mask, other=0.0).to(tl.float32)
    ga = (g * a).to(tl.bfloat16).to(tl.float32)
    h = (r + ga).to(tl.bfloat16)
    tl.store(HOUT + row * H_ + cols, h, mask=mask)

    x = h.to(tl.float32)
    mean = tl.sum(x, axis=0) / H_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / H_
    n = xc / tl.sqrt(var + eps)
    nb = n.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    t = (nb * w).to(tl.bfloat16).to(tl.float32)
    o = (t + b).to(tl.bfloat16)
    tl.store(OUT + row * H_ + cols, o, mask=mask)


@triton.jit
def _bias_gelu_kernel(X, B, OUT, N, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % K), mask=mask, other=0.0).to(tl.float32)
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    g = 0.5 * y * (1.0 + tl.extra.libdevice.erf(y * 0.7071067811865476))
    tl.store(OUT + offs, g.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_gated_res_kernel(X, B, RES, GATE, OUT, N, K: tl.constexpr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(GATE).to(tl.float32)
    g = tl.extra.libdevice.tanh(g).to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % K), mask=mask, other=0.0).to(tl.float32)
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    r = tl.load(RES + offs, mask=mask, other=0.0).to(tl.float32)
    gy = (g * y).to(tl.bfloat16).to(tl.float32)
    o = (r + gy).to(tl.bfloat16)
    tl.store(OUT + offs, o, mask=mask)


@torch.no_grad()
def run(
    hidden_state,
    input_layernorm_weight,
    input_layernorm_bias,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    post_attention_layernorm_weight,
    post_attention_layernorm_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    gate_attn,
    gate_ffn,
    norm_eps,
):
    bsz, seq_len, hidden_size = hidden_state.shape
    num_heads = 16
    head_dim = hidden_size // num_heads
    scaling = head_dim ** -0.5

    x = hidden_state.reshape(-1, hidden_size)
    M = x.shape[0]

    xn = torch.empty_like(x)
    _ln_kernel[(M,)](x, input_layernorm_weight, input_layernorm_bias, xn, M,
                     norm_eps, hidden_size, BLOCK_H, num_warps=8)

    q = torch.matmul(xn, q_proj_weight.t())
    k = torch.matmul(xn, k_proj_weight.t())
    v = torch.matmul(xn, v_proj_weight.t())

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    attn = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scaling)
    attn = attn.transpose(1, 2).reshape(M, hidden_size)
    attn = torch.matmul(attn, o_proj_weight.t())

    h2 = torch.empty_like(x)
    xn2 = torch.empty_like(x)
    _gated_res_ln_kernel[(M,)](
        x, attn, gate_attn, post_attention_layernorm_weight,
        post_attention_layernorm_bias, h2, xn2, M, norm_eps,
        hidden_size, BLOCK_H, num_warps=8)

    f1 = torch.matmul(xn2, fc1_weight.t())
    inter = fc1_weight.shape[0]
    Nf = M * inter
    g1 = torch.empty_like(f1)
    _bias_gelu_kernel[(triton.cdiv(Nf, 1024),)](f1, fc1_bias, g1, Nf, inter,
                                                1024, num_warps=4)

    f2 = torch.matmul(g1, fc2_weight.t())
    out = torch.empty_like(x)
    N2 = M * hidden_size
    _bias_gated_res_kernel[(triton.cdiv(N2, 1024),)](
        f2, fc2_bias, h2, gate_ffn, out, N2, hidden_size, 1024, num_warps=4)

    return out.view(bsz, seq_len, hidden_size)
