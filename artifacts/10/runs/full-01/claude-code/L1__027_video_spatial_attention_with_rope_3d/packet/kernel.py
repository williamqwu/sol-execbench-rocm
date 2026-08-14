import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem constants (fixed by definition.json)
# ---------------------------------------------------------------------------
HIDDEN = 1024
NHEAD = 16
HEAD_DIM = 64

# RoPE 3D block layout for head_dim=64:
#   temporal : dims [ 0, 21)  -> 10 rotated pairs, dim 20 passthrough
#   height   : dims [21, 42)  -> 10 rotated pairs, dim 41 passthrough
#   width    : dims [42, 63)  -> 10 rotated pairs, dim 62 passthrough
#   remaining: dim  63        -> passthrough
TEMPORAL_DIM = HEAD_DIM // 3                    # 21
SPATIAL_DIM = (HEAD_DIM - TEMPORAL_DIM) // 2    # 21
NPAIR = TEMPORAL_DIM // 2                       # 10


@triton.jit
def _rope_qkv_kernel(
    QKV, Q, K, V,
    COS_T, SIN_T, COS_H, SIN_H, COS_W, SIN_W,
    S, P, PPS,
    stride_qkv_b, stride_qkv_s,
    stride_o_b, stride_o_h, stride_o_s,
    SCALE_Q,
    BLOCK_S: tl.constexpr,
    NPAIR_C: tl.constexpr,
    TD: tl.constexpr,
    SD: tl.constexpr,
    HD: tl.constexpr,
    HID: tl.constexpr,
):
    """Fused split + 3D RoPE + layout transform.

    Reads the packed QKV projection [B, S, 3*HIDDEN] and writes contiguous
    Q, K, V of shape [B, NHEAD, S, HEAD_DIM], applying 3D RoPE to Q and K and
    folding the attention scale into Q.
    """
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s < S

    d = tl.arange(0, HD)

    # ---- positional indices ------------------------------------------------
    f_idx = s // P
    p_idx = s % P
    hh_idx = p_idx // PPS
    ww_idx = p_idx % PPS

    # ---- per-dim RoPE descriptor -------------------------------------------
    blk = tl.where(d < TD, 0, tl.where(d < TD + SD, 1, tl.where(d < TD + 2 * SD, 2, 3)))
    base = tl.where(d < TD, 0,
                    tl.where(d < TD + SD, TD,
                             tl.where(d < TD + 2 * SD, TD + SD, TD + 2 * SD)))
    local = d - base
    pair_i = local // 2
    even = (local % 2) == 0
    rotate = (blk < 3) & (local < 2 * NPAIR_C)
    pi = tl.minimum(pair_i, NPAIR_C - 1)
    partner = tl.where(even, d + 1, d - 1)

    # ---- cos/sin gather (tables precomputed in torch => bit-exact) ---------
    t_off = f_idx[:, None] * NPAIR_C + pi[None, :]
    h_off = hh_idx[:, None] * NPAIR_C + pi[None, :]
    w_off = ww_idx[:, None] * NPAIR_C + pi[None, :]
    m2 = s_mask[:, None] & rotate[None, :]

    ct = tl.load(COS_T + t_off, mask=m2, other=0.0)
    st = tl.load(SIN_T + t_off, mask=m2, other=0.0)
    ch = tl.load(COS_H + h_off, mask=m2, other=0.0)
    sh = tl.load(SIN_H + h_off, mask=m2, other=0.0)
    cw = tl.load(COS_W + w_off, mask=m2, other=0.0)
    sw = tl.load(SIN_W + w_off, mask=m2, other=0.0)

    bb = blk[None, :]
    cos_v = tl.where(bb == 0, ct, tl.where(bb == 1, ch, cw))
    sin_v = tl.where(bb == 0, st, tl.where(bb == 1, sh, sw))

    # ---- load q/k/v --------------------------------------------------------
    row = QKV + b * stride_qkv_b + s[:, None] * stride_qkv_s + h * HD
    q_self = tl.load(row + d[None, :], mask=s_mask[:, None], other=0.0)
    q_part = tl.load(row + partner[None, :], mask=s_mask[:, None], other=0.0)
    k_self = tl.load(row + HID + d[None, :], mask=s_mask[:, None], other=0.0)
    k_part = tl.load(row + HID + partner[None, :], mask=s_mask[:, None], other=0.0)
    v_val = tl.load(row + 2 * HID + d[None, :], mask=s_mask[:, None], other=0.0)

    # ---- rotation ----------------------------------------------------------
    # reference: r1 = x1*cos - x2*sin ; r2 = x1*sin + x2*cos
    # even lane holds r1 (x1=self, x2=partner); odd lane holds r2 (x1=partner, x2=self)
    q_rot = tl.where(even[None, :],
                     q_self * cos_v - q_part * sin_v,
                     q_part * sin_v + q_self * cos_v)
    q_out = tl.where(rotate[None, :], q_rot, q_self)

    k_rot = tl.where(even[None, :],
                     k_self * cos_v - k_part * sin_v,
                     k_part * sin_v + k_self * cos_v)
    k_out = tl.where(rotate[None, :], k_rot, k_self)

    # scale folded into Q; 1/sqrt(64) = 0.125 is an exact power of two, so this
    # multiply is bit-exact and commutes with the QK^T product.
    q_out = q_out * SCALE_Q

    obase = b * stride_o_b + h * stride_o_h + s[:, None] * stride_o_s + d[None, :]
    tl.store(Q + obase, q_out, mask=s_mask[:, None])
    tl.store(K + obase, k_out, mask=s_mask[:, None])
    tl.store(V + obase, v_val, mask=s_mask[:, None])


def _build_tables(num_frames, num_patches, pps, temporal_freqs, spatial_freqs):
    """cos/sin tables built with torch so they are bit-identical to the reference."""
    tf = temporal_freqs[:NPAIR]
    sf = spatial_freqs[:NPAIR]
    device = temporal_freqs.device

    n_h = (num_patches - 1) // pps + 1
    f_pos = torch.arange(num_frames, device=device, dtype=torch.float32)
    h_pos = torch.arange(n_h, device=device, dtype=torch.float32)
    w_pos = torch.arange(pps, device=device, dtype=torch.float32)

    ang_t = f_pos.unsqueeze(-1) * tf
    ang_h = h_pos.unsqueeze(-1) * sf
    ang_w = w_pos.unsqueeze(-1) * sf
    return (
        torch.cos(ang_t).contiguous(), torch.sin(ang_t).contiguous(),
        torch.cos(ang_h).contiguous(), torch.sin(ang_h).contiguous(),
        torch.cos(ang_w).contiguous(), torch.sin(ang_w).contiguous(),
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    temporal_freqs: torch.Tensor,
    spatial_freqs: torch.Tensor,
    scale: float,
):
    batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
    seq_len = num_frames * num_patches
    device = hidden_states.device

    pps = int(math.sqrt(num_patches))
    if pps * pps != num_patches:
        pps = int(math.ceil(math.sqrt(num_patches)))

    # ---- QKV projection (same rocBLAS call the reference makes) ------------
    qkv = F.linear(hidden_states.reshape(batch_size, seq_len, hidden_size),
                   qkv_weight, qkv_bias)

    cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = _build_tables(
        num_frames, num_patches, pps, temporal_freqs, spatial_freqs)

    q = torch.empty((batch_size, NHEAD, seq_len, HEAD_DIM),
                    device=device, dtype=torch.float32)
    k = torch.empty_like(q)
    v = torch.empty_like(q)

    BLOCK_S = 32
    grid = (triton.cdiv(seq_len, BLOCK_S), NHEAD, batch_size)
    _rope_qkv_kernel[grid](
        qkv, q, k, v,
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        seq_len, num_patches, pps,
        qkv.stride(0), qkv.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        scale,
        BLOCK_S=BLOCK_S,
        NPAIR_C=NPAIR,
        TD=TEMPORAL_DIM,
        SD=SPATIAL_DIM,
        HD=HEAD_DIM,
        HID=HIDDEN,
        num_warps=4,
        # The reference evaluates `x1*cos - x2*sin` as two separately-rounded
        # multiplies followed by an add. Letting LLVM contract that into an FMA
        # removes one rounding step and shifts results by ~1 ulp, which is
        # enough to miss this problem's (near bit-exact) tolerance.
        enable_fp_fusion=False,
    )

    # ---- attention: identical op sequence / reduction order to reference ----
    # rocBLAS is already at ~64 TF/s here and torch's softmax is at its
    # bandwidth limit; both are kept because any re-association (flash-style
    # online softmax, or a Triton exp) shifts results by >1 ulp and this
    # problem's tolerance is effectively bit-exact.
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    del attn_scores

    # Write the PV product straight into a [B, S, H, D] buffer through a
    # permuted view, so the head-major -> token-major transpose is absorbed by
    # the GEMM's output write instead of costing a separate full-size copy.
    attn_output = torch.empty((batch_size, seq_len, NHEAD, HEAD_DIM),
                              device=device, dtype=torch.float32)
    torch.matmul(attn_probs, v, out=attn_output.permute(0, 2, 1, 3))
    del attn_probs

    output = F.linear(attn_output.reshape(batch_size, seq_len, hidden_size),
                      out_weight, out_bias)
    return output.reshape(batch_size, num_frames, num_patches, hidden_size)
