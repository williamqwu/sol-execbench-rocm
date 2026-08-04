import torch
import torch.nn.functional as F

D_MODEL = 5120
NUM_HEADS = 20
HEAD_DIM = 256
SCALING = HEAD_DIM ** -0.5  # 0.0625, exact in bf16

_FA = None


def _get_fa():
    global _FA
    if _FA is None:
        try:
            from aiter import flash_attn_func
            _FA = flash_attn_func
        except Exception:
            _FA = False
    return _FA


def _attention(q, k, v, B, T):
    """q,k,v are (B*T, C) contiguous == (B, T, H, D) BSHD. Returns (B*T, C)."""
    fa = _get_fa()
    qb = q.view(B, T, NUM_HEADS, HEAD_DIM)
    kb = k.view(B, T, NUM_HEADS, HEAD_DIM)
    vb = v.view(B, T, NUM_HEADS, HEAD_DIM)
    if fa:
        o = fa(qb, kb, vb, softmax_scale=SCALING)
        if isinstance(o, (tuple, list)):
            o = o[0]
        return o.reshape(B * T, D_MODEL)
    qt = qb.transpose(1, 2)
    kt = kb.transpose(1, 2)
    vt = vb.transpose(1, 2)
    o = F.scaled_dot_product_attention(qt, kt, vt, scale=SCALING)
    return o.transpose(1, 2).reshape(B * T, D_MODEL)


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    embed_positions_weight: torch.Tensor,
    self_attn_layer_norm_weight: torch.Tensor,
    self_attn_layer_norm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    final_layer_norm_weight: torch.Tensor,
    final_layer_norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    B, Cin, T1 = input_features.shape
    T2 = T1 // 2
    C = D_MODEL
    M = B * T2
    dev = input_features.device
    dt = input_features.dtype

    # ---------- conv1 (k=3, pad=1) via im2col GEMM, output (B*T1, C) ----------
    padded = torch.zeros(B, T1 + 2, Cin, dtype=dt, device=dev)
    padded[:, 1:T1 + 1, :] = input_features.transpose(1, 2)
    im2col = padded.as_strided((B, T1, 3 * Cin), ((T1 + 2) * Cin, Cin, 1)).contiguous()
    w1 = conv1_weight.permute(0, 2, 1).reshape(C, 3 * Cin)
    g1 = torch._addmm_activation(conv1_bias, im2col.view(B * T1, 3 * Cin),
                                 w1.t(), use_gelu=True)
    del padded, im2col

    # ---------- conv2 (k=3, stride=2, pad=1) ----------
    # out[t] = odd[t-1]@W0^T + even[t]@W1^T + odd[t]@W2^T
    # g1 viewed as (B*T2, 2C) is exactly [even[t] | odd[t]]
    Wa = torch.cat([conv2_weight[:, :, 1], conv2_weight[:, :, 2]], dim=1)
    g1_2c = g1.view(M, 2 * C)
    y2 = F.linear(g1_2c, Wa, conv2_bias).view(B, T2, C)
    odd = g1.as_strided((M, C), (2 * C, 1), C)
    t0 = torch.mm(odd, conv2_weight[:, :, 0].t()).view(B, T2, C)
    del g1, g1_2c, odd, Wa
    y2[:, 1:, :] += t0[:, :-1, :]
    del t0
    hidden = F.gelu(y2)
    hidden += embed_positions_weight
    residual = hidden

    # ---------- self-attention ----------
    h = F.layer_norm(hidden, (C,), self_attn_layer_norm_weight,
                     self_attn_layer_norm_bias, 1e-5).view(M, C)

    # fused QKV: one wide GEMM (N=15360) is ~30% faster than three N=5120 GEMMs
    qkv_w = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv_b = torch.cat([q_proj_bias, torch.zeros_like(q_proj_bias), v_proj_bias])
    qkv = F.linear(h, qkv_w, qkv_b)
    del qkv_w, qkv_b, h
    q, k, v = qkv.split(C, dim=1)

    attn = _attention(q.contiguous(), k.contiguous(), v.contiguous(), B, T2)
    del qkv, q, k, v
    attn = F.linear(attn, out_proj_weight, out_proj_bias).view(B, T2, C)
    residual += attn
    del attn

    # ---------- FFN ----------
    h = F.layer_norm(residual, (C,), final_layer_norm_weight,
                     final_layer_norm_bias, 1e-5).view(M, C)
    h = torch._addmm_activation(fc1_bias, h, fc1_weight.t(), use_gelu=True)
    h = F.linear(h, fc2_weight, fc2_bias).view(B, T2, C)
    residual += h
    return residual
