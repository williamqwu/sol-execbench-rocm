import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _fused_bwd_kernel(
    grad_q_ptr, qpn_ptr, qnw_ptr, qrstd_ptr,
    grad_k_ptr, kpn_ptr, knw_ptr, krstd_ptr,
    grad_v_ptr,
    cos_ptr, sin_ptr,
    grad_qkv_ptr,
    grad_cos_ptr, grad_sin_ptr,
    grad_qnw_ptr, grad_knw_ptr,
    bsz, seq_len, num_heads, num_kv_heads, head_dim,
    q_size, kv_size, qkv_size,
    gq_b_s, gq_h_s, gq_s_s,
    gk_b_s, gk_h_s, gk_s_s,
    gv_b_s, gv_h_s, gv_s_s,
    cs_b_s, cs_s_s,
    qkv_b_s, qkv_s_s,
    qrstd_b_s, qrstd_h_s,
    krstd_b_s, krstd_h_s,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid % seq_len

    offs_lo = tl.arange(0, HALF)
    offs_hi = tl.arange(0, HALF) + HALF
    D = head_dim
    inv_D = 1.0 / D

    cs_base = b * cs_b_s + s * cs_s_s
    cos_lo = tl.load(cos_ptr + cs_base + offs_lo).to(tl.float32)
    cos_hi = tl.load(cos_ptr + cs_base + offs_hi).to(tl.float32)
    sin_lo = tl.load(sin_ptr + cs_base + offs_lo).to(tl.float32)
    sin_hi = tl.load(sin_ptr + cs_base + offs_hi).to(tl.float32)

    qw_lo = tl.load(qnw_ptr + offs_lo).to(tl.float32)
    qw_hi = tl.load(qnw_ptr + offs_hi).to(tl.float32)
    kw_lo = tl.load(knw_ptr + offs_lo).to(tl.float32)
    kw_hi = tl.load(knw_ptr + offs_hi).to(tl.float32)

    gcos_lo = tl.zeros([HALF], dtype=tl.float32)
    gcos_hi = tl.zeros([HALF], dtype=tl.float32)
    gsin_lo = tl.zeros([HALF], dtype=tl.float32)
    gsin_hi = tl.zeros([HALF], dtype=tl.float32)
    gqnw_lo = tl.zeros([HALF], dtype=tl.float32)
    gqnw_hi = tl.zeros([HALF], dtype=tl.float32)
    gknw_lo = tl.zeros([HALF], dtype=tl.float32)
    gknw_hi = tl.zeros([HALF], dtype=tl.float32)

    qkv_base = b * qkv_b_s + s * qkv_s_s

    # ============ Query path ============
    for h in range(num_heads):
        base = b * gq_b_s + h * gq_h_s + s * gq_s_s
        go_lo = tl.load(grad_q_ptr + base + offs_lo).to(tl.float32)
        go_hi = tl.load(grad_q_ptr + base + offs_hi).to(tl.float32)
        x_lo = tl.load(qpn_ptr + base + offs_lo).to(tl.float32)
        x_hi = tl.load(qpn_ptr + base + offs_hi).to(tl.float32)
        r = tl.load(qrstd_ptr + b * qrstd_b_s + h * qrstd_h_s + s).to(tl.float32)

        gqnw_lo += go_lo * x_lo * r
        gqnw_hi += go_hi * x_hi * r

        contrib_lo = go_lo * qw_lo * x_lo
        contrib_hi = go_hi * qw_hi * x_hi
        grad_rstd = tl.sum(contrib_lo, axis=0) + tl.sum(contrib_hi, axis=0)

        grad_pn_lo = go_lo * qw_lo * r + grad_rstd * (-r * r * r * x_lo * inv_D)
        grad_pn_hi = go_hi * qw_hi * r + grad_rstd * (-r * r * r * x_hi * inv_D)

        gp_lo = grad_pn_lo.to(tl.bfloat16).to(tl.float32)
        gp_hi = grad_pn_hi.to(tl.bfloat16).to(tl.float32)

        gpr_lo = gp_lo * cos_lo + gp_hi * sin_lo
        gpr_hi = gp_hi * cos_hi - gp_lo * sin_hi

        q_off = qkv_base + h * D
        tl.store(grad_qkv_ptr + q_off + offs_lo, gpr_lo.to(tl.bfloat16))
        tl.store(grad_qkv_ptr + q_off + offs_hi, gpr_hi.to(tl.bfloat16))

        xo_lo = x_lo * cos_lo - x_hi * sin_lo
        xo_hi = x_hi * cos_hi + x_lo * sin_hi
        xo_lo_bf = xo_lo.to(tl.bfloat16).to(tl.float32)
        xo_hi_bf = xo_hi.to(tl.bfloat16).to(tl.float32)

        gcos_lo += gp_lo * xo_lo_bf
        gcos_hi += gp_hi * xo_hi_bf
        gsin_lo -= gp_lo * xo_hi_bf
        gsin_hi += gp_hi * xo_lo_bf

    # ============ Key path ============
    for h in range(num_kv_heads):
        base = b * gk_b_s + h * gk_h_s + s * gk_s_s
        go_lo = tl.load(grad_k_ptr + base + offs_lo).to(tl.float32)
        go_hi = tl.load(grad_k_ptr + base + offs_hi).to(tl.float32)
        x_lo = tl.load(kpn_ptr + base + offs_lo).to(tl.float32)
        x_hi = tl.load(kpn_ptr + base + offs_hi).to(tl.float32)
        r = tl.load(krstd_ptr + b * krstd_b_s + h * krstd_h_s + s).to(tl.float32)

        gknw_lo += go_lo * x_lo * r
        gknw_hi += go_hi * x_hi * r

        contrib_lo = go_lo * kw_lo * x_lo
        contrib_hi = go_hi * kw_hi * x_hi
        grad_rstd = tl.sum(contrib_lo, axis=0) + tl.sum(contrib_hi, axis=0)

        grad_pn_lo = go_lo * kw_lo * r + grad_rstd * (-r * r * r * x_lo * inv_D)
        grad_pn_hi = go_hi * kw_hi * r + grad_rstd * (-r * r * r * x_hi * inv_D)

        gp_lo = grad_pn_lo.to(tl.bfloat16).to(tl.float32)
        gp_hi = grad_pn_hi.to(tl.bfloat16).to(tl.float32)

        gpr_lo = gp_lo * cos_lo + gp_hi * sin_lo
        gpr_hi = gp_hi * cos_hi - gp_lo * sin_hi

        k_off = qkv_base + q_size + h * D
        tl.store(grad_qkv_ptr + k_off + offs_lo, gpr_lo.to(tl.bfloat16))
        tl.store(grad_qkv_ptr + k_off + offs_hi, gpr_hi.to(tl.bfloat16))

        xo_lo = x_lo * cos_lo - x_hi * sin_lo
        xo_hi = x_hi * cos_hi + x_lo * sin_hi
        xo_lo_bf = xo_lo.to(tl.bfloat16).to(tl.float32)
        xo_hi_bf = xo_hi.to(tl.bfloat16).to(tl.float32)

        gcos_lo += gp_lo * xo_lo_bf
        gcos_hi += gp_hi * xo_hi_bf
        gsin_lo -= gp_lo * xo_hi_bf
        gsin_hi += gp_hi * xo_lo_bf

    # ============ Value path (just transpose) ============
    for h in range(num_kv_heads):
        base = b * gv_b_s + h * gv_h_s + s * gv_s_s
        gv_lo = tl.load(grad_v_ptr + base + offs_lo)
        gv_hi = tl.load(grad_v_ptr + base + offs_hi)
        v_off = qkv_base + q_size + kv_size + h * D
        tl.store(grad_qkv_ptr + v_off + offs_lo, gv_lo)
        tl.store(grad_qkv_ptr + v_off + offs_hi, gv_hi)

    tl.store(grad_cos_ptr + cs_base + offs_lo, gcos_lo.to(tl.bfloat16))
    tl.store(grad_cos_ptr + cs_base + offs_hi, gcos_hi.to(tl.bfloat16))
    tl.store(grad_sin_ptr + cs_base + offs_lo, gsin_lo.to(tl.bfloat16))
    tl.store(grad_sin_ptr + cs_base + offs_hi, gsin_hi.to(tl.bfloat16))

    tl.atomic_add(grad_qnw_ptr + offs_lo, gqnw_lo)
    tl.atomic_add(grad_qnw_ptr + offs_hi, gqnw_hi)
    tl.atomic_add(grad_knw_ptr + offs_lo, gknw_lo)
    tl.atomic_add(grad_knw_ptr + offs_hi, gknw_hi)


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
    eps: float
):
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    bsz, seq_len, hidden_size = hidden_states.shape
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size

    grad_qkv = torch.empty(bsz, seq_len, qkv_size, dtype=torch.bfloat16, device=hidden_states.device)
    grad_cos = torch.empty(bsz, seq_len, head_dim, dtype=torch.bfloat16, device=hidden_states.device)
    grad_sin = torch.empty(bsz, seq_len, head_dim, dtype=torch.bfloat16, device=hidden_states.device)
    grad_qnw = torch.zeros(head_dim, dtype=torch.float32, device=hidden_states.device)
    grad_knw = torch.zeros(head_dim, dtype=torch.float32, device=hidden_states.device)

    qrstd = q_rstd.squeeze(-1)
    krstd = k_rstd.squeeze(-1)

    grid = (bsz * seq_len,)
    _fused_bwd_kernel[grid](
        grad_query, query_pre_norm, q_norm_weight, qrstd,
        grad_key, key_pre_norm, k_norm_weight, krstd,
        grad_value,
        cos, sin,
        grad_qkv,
        grad_cos, grad_sin,
        grad_qnw, grad_knw,
        bsz, seq_len, num_heads, num_kv_heads, head_dim,
        q_size, kv_size, qkv_size,
        grad_query.stride(0), grad_query.stride(1), grad_query.stride(2),
        grad_key.stride(0), grad_key.stride(1), grad_key.stride(2),
        grad_value.stride(0), grad_value.stride(1), grad_value.stride(2),
        cos.stride(0), cos.stride(1),
        grad_qkv.stride(0), grad_qkv.stride(1),
        qrstd.stride(0), qrstd.stride(1),
        krstd.stride(0), krstd.stride(1),
        HALF=head_dim // 2,
    )

    grad_qkv_flat = grad_qkv.reshape(-1, qkv_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    grad_hidden_states = F.linear(grad_qkv_flat, qkv_weight.t())
    grad_hidden_states = grad_hidden_states.reshape(bsz, seq_len, hidden_size)
    grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), hidden_flat)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_qkv_weight.to(torch.bfloat16),
        grad_qnw.to(torch.bfloat16),
        grad_knw.to(torch.bfloat16)
    )
