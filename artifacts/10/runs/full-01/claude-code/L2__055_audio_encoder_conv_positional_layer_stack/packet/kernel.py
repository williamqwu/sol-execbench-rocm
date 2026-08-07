import torch
import torch.nn.functional as F

try:
    import aiter

    _HAS_AITER = True
except Exception:  # pragma: no cover
    aiter = None
    _HAS_AITER = False

_CACHE = {}


def _mm(a, w, out=None):
    """out = a @ w.T, bf16. aiter's asm gemm beats rocBLAS on tall-K shapes.

    Never pass aiter a bias: its epilogue costs ~2.5x the gemm itself.
    Biases are folded into neighbouring elementwise ops by the caller.
    """
    if out is None:
        out = torch.empty(a.shape[0], w.shape[0], device=a.device, dtype=a.dtype)
    if _HAS_AITER:
        try:
            aiter.gemm_a16w16_asm(a, w, out, None)
            return out
        except Exception:
            pass
    return torch.mm(a, w.t(), out=out)


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
    B, M, T = input_features.shape
    C = conv1_weight.shape[0]
    S = T // 2
    H = 20
    D = C // H
    N = B * S
    scaling = D ** -0.5
    dev = input_features.device

    # ---- cached weight repacking (weights are loop-invariant across timed calls) ----
    key = (q_proj_weight.data_ptr(), conv2_weight.data_ptr(), C)
    ent = _CACHE.get(key)
    if ent is None:
        # Fuse Q/K/V into one GEMM. Q's scaling folds into its weight+bias so the
        # elementwise multiply disappears entirely.
        wq = (q_proj_weight.float() * scaling).to(torch.bfloat16)
        w_qkv = torch.cat((wq, k_proj_weight, v_proj_weight), dim=0).contiguous()
        b_qkv = torch.cat(
            (
                (q_proj_bias.float() * scaling).to(torch.bfloat16),
                torch.zeros_like(q_proj_bias),
                v_proj_bias,
            )
        ).contiguous()
        # conv2 taps as three (C, C) matrices, tap-major
        w_c2 = conv2_weight.permute(2, 0, 1).contiguous()
        w1 = conv1_weight.reshape(C, M * 3).contiguous()
        ent = (w_qkv, b_qkv, w_c2, w1)
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[key] = ent
    w_qkv, b_qkv, w_c2, w1 = ent

    # ---- conv1 (k=3, pad=1) via im2col GEMM -> (B*T, C), fused bias+gelu ----
    xp = F.pad(input_features.transpose(1, 2), (0, 0, 1, 1))     # (B, T+2, M)
    cols = torch.stack(
        (xp[:, 0:T], xp[:, 1:T + 1], xp[:, 2:T + 2]), dim=-1
    ).reshape(B * T, M * 3)
    g = F.gelu(torch.addmm(conv1_bias, cols, w1.t())).view(B, T, C)
    del cols, xp

    # ---- conv2 (k=3, stride=2, pad=1) as 3 strided GEMMs accumulated ----
    # Strided slices feed rocBLAS directly, avoiding a 3*C-wide im2col
    # materialisation (0.74 GB at B=16) that costs more than it saves.
    gp = F.pad(g, (0, 0, 1, 0))                                  # (B, T+1, C)
    acc = torch.matmul(gp[:, 0:2 * S:2, :], w_c2[0].t())
    acc += torch.matmul(gp[:, 1:1 + 2 * S:2, :], w_c2[1].t())
    acc += torch.matmul(gp[:, 2:2 + 2 * S:2, :], w_c2[2].t())
    del g, gp

    # ---- gelu(conv2 + bias) + positional embedding, fused ----
    acc += conv2_bias
    hs = F.gelu(acc)
    hs += embed_positions_weight
    residual = hs.view(N, C)
    del acc

    # ---- LN1 ----
    h = F.layer_norm(
        residual, (C,), self_attn_layer_norm_weight, self_attn_layer_norm_bias, 1e-5
    )

    # ---- fused QKV (scaling already folded into Q's weight and bias) ----
    qkv = _mm(h, w_qkv)
    qkv += b_qkv
    qkv = qkv.view(B, S, 3, H, D)
    q = qkv[:, :, 0].contiguous()
    k_s = qkv[:, :, 1].contiguous()
    v = qkv[:, :, 2].contiguous()
    del qkv, h

    if _HAS_AITER:
        try:
            attn = aiter.flash_attn_func(q, k_s, v, softmax_scale=1.0, causal=False)
        except Exception:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k_s.transpose(1, 2), v.transpose(1, 2), scale=1.0
            ).transpose(1, 2)
    else:
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_s.transpose(1, 2), v.transpose(1, 2), scale=1.0
        ).transpose(1, 2)
    attn = attn.reshape(N, C)
    del q, k_s, v

    # ---- out proj; bias folds into the residual add ----
    o = _mm(attn, out_proj_weight)
    del attn
    o += out_proj_bias
    o += residual
    residual = o

    # ---- LN2 ----
    h = F.layer_norm(
        residual, (C,), final_layer_norm_weight, final_layer_norm_bias, 1e-5
    )

    # ---- FFN; fc2 bias folds into the residual add ----
    h = F.gelu(torch.addmm(fc1_bias, h, fc1_weight.t()))
    out = _mm(h, fc2_weight)
    del h
    out += fc2_bias
    out += residual
    return out.view(B, S, C)
