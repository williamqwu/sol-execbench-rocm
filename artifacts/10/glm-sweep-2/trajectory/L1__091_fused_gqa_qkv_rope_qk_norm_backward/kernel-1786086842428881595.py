import torch
import torch.nn.functional as F
from typing import Tuple
import triton
import triton.language as tl


@triton.jit
def _fused_qk_bwd_kernel(
    grad_out_ptr,       # [B, H, S, D]
    x_pre_norm_ptr,     # [B, H, S, D]
    cos_ptr,            # [B, S, D]
    sin_ptr,            # [B, S, D]
    weight_ptr,         # [D]
    rstd_ptr,           # [B, H, S, 1]
    out_ptr,            # [B, S, H*D]  transposed output
    grad_cos_ptr,       # [B, S, D] fp32
    grad_sin_ptr,       # [B, S, D] fp32
    grad_w_part_ptr,    # [B, S, D] partial grad_weight (fp32)
    B, S,
    gout_b, gout_h, gout_s,
    xpn_b, xpn_h, xpn_s,
    cs_b, cs_s,
    rst_b, rst_h, rst_s,
    out_b, out_s,
    gcs_b, gcs_s,
    H: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    ACCUMULATE: tl.constexpr,  # if True, atomic_add to grad_cos/sin; else store
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    d_offs = tl.arange(0, D)           # [D]
    h_offs = tl.arange(0, HALF)        # [HALF]

    cs_base = pid_b * cs_b + pid_s * cs_s
    cos = tl.load(cos_ptr + cs_base + d_offs).to(tl.float32)
    sin = tl.load(sin_ptr + cs_base + d_offs).to(tl.float32)

    cos1 = cos[:HALF]
    cos2 = cos[HALF:]
    sin1 = sin[:HALF]
    sin2 = sin[HALF:]

    w = tl.load(weight_ptr + d_offs).to(tl.float32)
    w1 = w[:HALF]
    w2 = w[HALF:]

    inv_D = 1.0 / D

    acc_gc1 = tl.zeros([HALF], dtype=tl.float32)
    acc_gc2 = tl.zeros([HALF], dtype=tl.float32)
    acc_gs1 = tl.zeros([HALF], dtype=tl.float32)
    acc_gs2 = tl.zeros([HALF], dtype=tl.float32)
    acc_gw = tl.zeros([D], dtype=tl.float32)

    for h in range(H):
        base = pid_b * gout_b + h * gout_h + pid_s * gout_s
        xbase = pid_b * xpn_b + h * xpn_h + pid_s * xpn_s

        go = tl.load(grad_out_ptr + base + d_offs).to(tl.float32)
        xp = tl.load(x_pre_norm_ptr + xbase + d_offs).to(tl.float32)
        rst = tl.load(rstd_ptr + pid_b * rst_b + h * rst_h + pid_s * rst_s).to(tl.float32)

        # --- RMS norm backward ---
        x_normed = xp * rst
        acc_gw += go * x_normed

        gw_rf = w * rst
        grad_x_direct = go * gw_rf
        grad_rstd = tl.sum(go * w * xp)
        grad_x = grad_x_direct + grad_rstd * (-(rst * rst * rst) * xp * inv_D)

        gx1 = grad_x[:HALF]
        gx2 = grad_x[HALF:]

        # --- RoPE backward ---
        # grad_rotated_inv = cat(gx2, -gx1)
        # grad_pre_rope = grad_x * cos + grad_rotated_inv * sin
        gpr1 = gx1 * cos1 + gx2 * sin1
        gpr2 = gx2 * cos2 + (-gx1) * sin2

        out_base = pid_b * out_b + pid_s * out_s + h * D
        tl.store(out_ptr + out_base + h_offs, gpr1.to(tl.bfloat16))
        tl.store(out_ptr + out_base + HALF + h_offs, gpr2.to(tl.bfloat16))

        # --- grad_cos, grad_sin ---
        xp1 = xp[:HALF]
        xp2 = xp[HALF:]
        # x_original = x_rotated * cos + x_rotated_inv * sin
        # x_rotated_inv = cat(-xp2, xp1)
        xo1 = xp1 * cos1 + (-xp2) * sin1
        xo2 = xp2 * cos2 + xp1 * sin2

        go1 = go[:HALF]
        go2 = go[HALF:]

        acc_gc1 += go1 * xo1
        acc_gc2 += go2 * xo2

        # x_original_rotated = cat(-xo2, xo1)
        acc_gs1 += go1 * (-xo2)
        acc_gs2 += go2 * xo1

    # Write grad_cos, grad_sin
    gcs_base = pid_b * gcs_b + pid_s * gcs_s
    gc1_val = acc_gc1
    gc2_val = acc_gc2
    gs1_val = acc_gs1
    gs2_val = acc_gs2
    if ACCUMULATE:
        tl.atomic_add(grad_cos_ptr + gcs_base + h_offs, gc1_val)
        tl.atomic_add(grad_cos_ptr + gcs_base + HALF + h_offs, gc2_val)
        tl.atomic_add(grad_sin_ptr + gcs_base + h_offs, gs1_val)
        tl.atomic_add(grad_sin_ptr + gcs_base + HALF + h_offs, gs2_val)
    else:
        tl.store(grad_cos_ptr + gcs_base + h_offs, gc1_val)
        tl.store(grad_cos_ptr + gcs_base + HALF + h_offs, gc2_val)
        tl.store(grad_sin_ptr + gcs_base + h_offs, gs1_val)
        tl.store(grad_sin_ptr + gcs_base + HALF + h_offs, gs2_val)

    # Write grad_weight partial
    gwp_base = pid_b * (S * D) + pid_s * D
    tl.store(grad_w_part_ptr + gwp_base + h_offs, acc_gw[:HALF])
    tl.store(grad_w_part_ptr + gwp_base + HALF + h_offs, acc_gw[HALF:])


@triton.jit
def _transpose_v_kernel(
    grad_value_ptr,  # [B, H, S, D]
    out_ptr,          # [B, S, H*D]
    B, S,
    gv_b, gv_h, gv_s,
    out_b, out_s,
    H: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    d_offs = tl.arange(0, D)
    out_base = pid_b * out_b + pid_s * out_s
    for h in range(H):
        base = pid_b * gv_b + h * gv_h + pid_s * gv_s
        gv = tl.load(grad_value_ptr + base + d_offs)
        tl.store(out_ptr + out_base + h * D + d_offs, gv)


@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_pre_norm: torch.Tensor,
    key_pre_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    eps: float,
):
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    bsz, seq_len, hidden_size = hidden_states.shape
    qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim  # 6144

    q_part = num_heads * head_dim       # 4096
    k_part = num_kv_heads * head_dim    # 1024
    v_part = num_kv_heads * head_dim    # 1024
    HALF = head_dim // 2

    dev = hidden_states.device

    grad_qkv_states = torch.empty(bsz, seq_len, qkv_size, dtype=torch.bfloat16, device=dev)
    grad_cos_f = torch.zeros(bsz, seq_len, head_dim, dtype=torch.float32, device=dev)
    grad_sin_f = torch.zeros(bsz, seq_len, head_dim, dtype=torch.float32, device=dev)
    gw_q_part = torch.empty(bsz, seq_len, head_dim, dtype=torch.float32, device=dev)
    gw_k_part = torch.empty(bsz, seq_len, head_dim, dtype=torch.float32, device=dev)

    grid = (bsz, seq_len)

    # Q path
    _fused_qk_bwd_kernel[grid](
        grad_query, query_pre_norm, cos, sin, q_norm_weight, q_rstd,
        grad_qkv_states[:, :, :q_part],
        grad_cos_f, grad_sin_f, gw_q_part,
        bsz, seq_len,
        grad_query.stride(0), grad_query.stride(1), grad_query.stride(2),
        query_pre_norm.stride(0), query_pre_norm.stride(1), query_pre_norm.stride(2),
        cos.stride(0), cos.stride(1),
        q_rstd.stride(0), q_rstd.stride(1), q_rstd.stride(2),
        grad_qkv_states.stride(0), grad_qkv_states.stride(1),
        grad_cos_f.stride(0), grad_cos_f.stride(1),
        H=num_heads, D=head_dim, HALF=HALF, ACCUMULATE=False,
    )

    # K path (accumulate into grad_cos_f, grad_sin_f)
    _fused_qk_bwd_kernel[grid](
        grad_key, key_pre_norm, cos, sin, k_norm_weight, k_rstd,
        grad_qkv_states[:, :, q_part:q_part + k_part],
        grad_cos_f, grad_sin_f, gw_k_part,
        bsz, seq_len,
        grad_key.stride(0), grad_key.stride(1), grad_key.stride(2),
        key_pre_norm.stride(0), key_pre_norm.stride(1), key_pre_norm.stride(2),
        cos.stride(0), cos.stride(1),
        k_rstd.stride(0), k_rstd.stride(1), k_rstd.stride(2),
        grad_qkv_states.stride(0), grad_qkv_states.stride(1),
        grad_cos_f.stride(0), grad_cos_f.stride(1),
        H=num_kv_heads, D=head_dim, HALF=HALF, ACCUMULATE=True,
    )

    # V path: transpose
    _transpose_v_kernel[grid](
        grad_value,
        grad_qkv_states[:, :, q_part + k_part:],
        bsz, seq_len,
        grad_value.stride(0), grad_value.stride(1), grad_value.stride(2),
        grad_qkv_states.stride(0), grad_qkv_states.stride(1),
        H=num_kv_heads, D=head_dim,
    )

    # Reduce grad_weight partials
    grad_q_norm_weight = gw_q_part.sum(dim=(0, 1)).to(torch.bfloat16)
    grad_k_norm_weight = gw_k_part.sum(dim=(0, 1)).to(torch.bfloat16)

    grad_cos = grad_cos_f.to(torch.bfloat16)
    grad_sin = grad_sin_f.to(torch.bfloat16)

    # GEMMs
    grad_qkv_flat = grad_qkv_states.reshape(-1, qkv_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)

    grad_hidden_states = torch.mm(grad_qkv_flat, qkv_weight).reshape(bsz, seq_len, hidden_size)
    grad_qkv_weight = torch.mm(grad_qkv_flat.t(), hidden_flat)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_cos,
        grad_sin,
        grad_qkv_weight.to(torch.bfloat16),
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
