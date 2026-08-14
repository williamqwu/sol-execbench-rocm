import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reverse transpose + reverse reshape:  (B, H, S, D) -> (B*S, H*D)
#
# Replaces the reference's `.transpose(1,2).contiguous().view(...)`.
# Indexing is flat over the *output* so each program writes one fully
# contiguous BLK-element run; the gather side stays 1 KiB-contiguous per
# (m, h) because D is the fastest axis on both sides.  ~20% faster than
# torch's transpose+contiguous.
# ---------------------------------------------------------------------------
@triton.jit
def _permute_qkv(
    SRC,                 # (B, H, S, D) contiguous
    OUT,                 # (B*S, H*D) row-major
    S,
    TOT,                 # M * H * D
    s_b, s_h, s_s,       # src strides (elements)
    HD: tl.constexpr,    # H * D
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    mask = off < TOT

    m = off // HD
    r = off % HD
    h = r // D
    d = r % D

    b = m // S
    s = m % S

    val = tl.load(SRC + (b * s_b + h * s_h + s * s_s + d), mask=mask, other=0.0)
    tl.store(OUT + off, val, mask=mask)


# ---------------------------------------------------------------------------
# GQA un-repeat fused with reverse-transpose + reverse-reshape, for K and V
# together in one launch and one pass over each input.
#
# Reference, per tensor:
#     x.view(B, KV, G, S, D).sum(dim=2)          -> head h = kv*G + g
#      .transpose(1, 2).contiguous().view(B, S, KV*D)
#
# That is three kernels and two full-size temporaries per tensor (six and
# four for K and V together).  This is one kernel and no temporaries.  The
# G-accumulation runs left-to-right, (((g0+g1)+g2)+g3), which is the order
# torch's sum(dim=2) uses over a length-4 contiguous-stride axis, so the
# result is bit-identical to the reference.
# ---------------------------------------------------------------------------
@triton.jit
def _gqa_sum_permute2(
    SRC_K,               # (B, KV*G, S, D) contiguous
    SRC_V,               # (B, KV*G, S, D) contiguous
    OUT,                 # (2, B*S, KV*D) contiguous
    S,
    TOT,                 # M * KV * D
    s_b, s_h, s_s,       # strides shared by both sources
    out_plane_stride,
    KVD: tl.constexpr,   # KV * D
    G: tl.constexpr,
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    mask = off < TOT

    m = off // KVD
    r = off % KVD
    kv = r // D
    d = r % D

    b = m // S
    s = m % S

    base = b * s_b + s * s_s + d + (kv * G) * s_h

    ak = tl.load(SRC_K + base, mask=mask, other=0.0)
    for g in tl.static_range(1, G):
        ak += tl.load(SRC_K + base + g * s_h, mask=mask, other=0.0)
    tl.store(OUT + off, ak, mask=mask)

    av = tl.load(SRC_V + base, mask=mask, other=0.0)
    for g in tl.static_range(1, G):
        av += tl.load(SRC_V + base + g * s_h, mask=mask, other=0.0)
    tl.store(OUT + out_plane_stride + off, av, mask=mask)


# Tuned on MI355X by sweeping (BLK, num_warps) over every workload shape.
_PERM_BLK, _PERM_NW = 512, 2
_GQA_BLK, _GQA_NW = 256, 4


def _launch(tot, blk):
    return (triton.cdiv(tot, blk),)


@torch.no_grad()
def run(
    grad_query_states: torch.Tensor,
    grad_key_states: torch.Tensor,
    grad_value_states: torch.Tensor,
    decoder_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    num_heads = grad_query_states.shape[1]                 # 16
    head_dim = grad_query_states.shape[3]                  # 256
    num_key_value_heads = k_weight.shape[0] // head_dim    # 4
    G = num_heads // num_key_value_heads                   # 4

    B, Sd, hidden = decoder_hidden_states.shape            # (*, *, 1536)
    Se = encoder_hidden_states.shape[1]
    cross = encoder_hidden_states.shape[2]                 # 1024

    q_out = num_heads * head_dim                           # 4096
    kv_out = num_key_value_heads * head_dim                # 1024

    dev = grad_query_states.device
    M_dec = B * Sd
    M_enc = B * Se

    gq = grad_query_states.contiguous()
    gk = grad_key_states.contiguous()
    gv = grad_value_states.contiguous()

    # ------------------------- query path -------------------------
    gqp = torch.empty((M_dec, q_out), dtype=torch.float32, device=dev)
    tot = M_dec * q_out
    _permute_qkv[_launch(tot, _PERM_BLK)](
        gq, gqp,
        Sd, tot,
        gq.stride(0), gq.stride(1), gq.stride(2),
        HD=q_out, D=head_dim, BLK=_PERM_BLK,
        num_warps=_PERM_NW,
    )

    dec_flat = decoder_hidden_states.reshape(M_dec, hidden)

    # The reference's matmul(3-D, 2-D) folds to exactly these mm's, so the
    # K-reduction order -- and hence every rounding step -- is identical.
    # (Any re-blocked or split-precision GEMM here lands ~20x outside the
    # stated tolerance: it is tight enough that even exact arithmetic
    # differs from hipBLASLt's fp32 result on most elements.)
    grad_decoder_hidden_states = torch.mm(gqp, q_weight).view(B, Sd, hidden)
    grad_q_weight = torch.mm(gqp.t(), dec_flat)

    # ----------------------- key / value path ---------------------
    # One allocation with two contiguous planes: kvp[0] == grad_key_proj and
    # kvp[1] == grad_value_proj, each a genuine row-major (M_enc, kv_out)
    # tensor, so the GEMMs below receive exactly the reference's operands.
    kvp = torch.empty((2, M_enc, kv_out), dtype=torch.float32, device=dev)
    tot = M_enc * kv_out
    _gqa_sum_permute2[_launch(tot, _GQA_BLK)](
        gk, gv, kvp,
        Se, tot,
        gk.stride(0), gk.stride(1), gk.stride(2),
        kvp.stride(0),
        KVD=kv_out, G=G, D=head_dim, BLK=_GQA_BLK,
        num_warps=_GQA_NW,
    )
    gkp = kvp[0]
    gvp = kvp[1]

    enc_flat = encoder_hidden_states.reshape(M_enc, cross)

    # grad_enc = fl( fl(gkp @ kw) + fl(gvp @ vw) ) -- addmm fuses the add
    # into the second GEMM's epilogue, saving a full read-modify-write pass
    # over (M_enc x 1024) without changing the arithmetic.
    ge = torch.mm(gkp, k_weight)
    ge = torch.addmm(ge, gvp, v_weight)
    grad_encoder_hidden_states = ge.view(B, Se, cross)

    grad_k_weight = torch.mm(gkp.t(), enc_flat)
    grad_v_weight = torch.mm(gvp.t(), enc_flat)

    return (
        grad_decoder_hidden_states,
        grad_encoder_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
    )
