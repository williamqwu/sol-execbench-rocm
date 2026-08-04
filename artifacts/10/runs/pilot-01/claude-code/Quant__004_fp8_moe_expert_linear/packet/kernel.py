"""FP8-quantized MoE expert (gate-up -> SiLU*up -> down -> routing) for MI355X.

Semantics follow reference.py exactly:
  * activations quantized BlockWise1x128, weights BlockWise128x128
  * scale = clamp(amax/448, min=1e-12); q = clamp(x/scale, +-448) -> e4m3fn
  * GEMM accumulates in fp32, result rounded to bf16
  * SiLU and the elementwise product happen in bf16 (fp32 opmath, bf16 rounding)
  * routing weight applied to the bf16 down-projection output, rounded to bf16

Fusions vs. the reference:
  * gate and up column tiles are computed by the same program, so the
    (M, 2*intermediate) bf16 intermediate is never materialized; SiLU, the
    product and the 1x128 quantization of the result all happen in registers.
  * both weight quantizations run in a single launch.
  * the routing weight is folded into the second GEMM's epilogue.
"""

import torch
import triton
import triton.language as tl

E4M3_MAX = tl.constexpr(448.0)
# torch on GPU folds `x / 448.0` (python scalar) into a multiply by the fp32
# reciprocal; reproduce that exactly, or fp8 rounding ties flip.
E4M3_RCP = tl.constexpr(float(__import__('numpy').float32(1.0) / __import__('numpy').float32(448.0)))
SMIN = tl.constexpr(1e-12)
FP8 = tl.constexpr(tl.float8e4nv)
E4M3_MAX_PY = 448.0
_FMAX: tl.constexpr = 448.0
_FMIN: tl.constexpr = 1e-12


# --------------------------------------------------------------------------
# activation quantization: BlockWise1x128
# --------------------------------------------------------------------------
@triton.jit
def _quant_act_kernel(
    X, Q, S,
    M, stride_xm, stride_qm, stride_sm,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * 128 + tl.arange(0, 128)
    m = rm < M
    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :],
                mask=m[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    s = tl.maximum(amax * E4M3_RCP, SMIN)
    q = tl.fdiv(x, s[:, None], ieee_rounding=True)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    tl.store(Q + rm[:, None] * stride_qm + rk[None, :], q.to(FP8, fp_downcast_rounding="rtne"), mask=m[:, None])
    tl.store(S + rm * stride_sm + pid_k, s, mask=m)


# --------------------------------------------------------------------------
# weight quantization: BlockWise128x128, both weights in one launch
# --------------------------------------------------------------------------
@triton.jit
def _quant_w2_kernel(
    W0, Q0, S0, stride_w0, stride_q0, stride_s0,
    W1, Q1, S1, stride_w1, stride_q1, stride_s1,
    NK0: tl.constexpr, NTILES0: tl.constexpr, NK1: tl.constexpr,
):
    pid = tl.program_id(0)
    rr = tl.arange(0, 128)
    if pid < NTILES0:
        pid_n = pid // NK0
        pid_k = pid % NK0
        rn = pid_n * 128 + rr
        rk = pid_k * 128 + rr
        w = tl.load(W0 + rn[:, None] * stride_w0 + rk[None, :]).to(tl.float32)
        amax = tl.max(tl.abs(w))
        s = tl.maximum(amax * E4M3_RCP, SMIN)
        q = tl.minimum(tl.maximum(tl.fdiv(w, s, ieee_rounding=True), -E4M3_MAX), E4M3_MAX)
        tl.store(Q0 + rn[:, None] * stride_q0 + rk[None, :], q.to(FP8, fp_downcast_rounding="rtne"))
        tl.store(S0 + pid_n * stride_s0 + pid_k, s)
    else:
        p = pid - NTILES0
        pid_n = p // NK1
        pid_k = p % NK1
        rn = pid_n * 128 + rr
        rk = pid_k * 128 + rr
        w = tl.load(W1 + rn[:, None] * stride_w1 + rk[None, :]).to(tl.float32)
        amax = tl.max(tl.abs(w))
        s = tl.maximum(amax * E4M3_RCP, SMIN)
        q = tl.minimum(tl.maximum(tl.fdiv(w, s, ieee_rounding=True), -E4M3_MAX), E4M3_MAX)
        tl.store(Q1 + rn[:, None] * stride_q1 + rk[None, :], q.to(FP8, fp_downcast_rounding="rtne"))
        tl.store(S1 + pid_n * stride_s1 + pid_k, s)


# --------------------------------------------------------------------------
# GEMM 1 (gate & up) fused with SiLU, product and 1x128 re-quantization
# --------------------------------------------------------------------------
@triton.jit
def _gemm1_kernel(
    A, SA, B, SB, GQ, GS,
    M,
    stride_am, stride_sam, stride_bn, stride_sbn, stride_gm, stride_gsm,
    KBLK: tl.constexpr,      # K // 128
    NTILE: tl.constexpr,     # intermediate // 128
    IHALF: tl.constexpr,     # intermediate (row offset of the "up" half of B)
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_MT: tl.constexpr,
):
    pid = tl.program_id(0)
    # grouped ordering along M so B tiles get reused out of L2
    width = GROUP_M * NTILE
    gid = pid // width
    off = pid % width
    first = gid * GROUP_M
    gsz = min(NUM_MT - first, GROUP_M)
    pid_m = first + (off % gsz)
    pid_n = off // gsz

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    ram = tl.where(rmask, rm, 0)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)

    a_ptr = A + ram[:, None] * stride_am + rk[None, :]
    bg_ptr = B + (pid_n * 128 + rn)[:, None] * stride_bn + rk[None, :]
    bu_ptr = B + (IHALF + pid_n * 128 + rn)[:, None] * stride_bn + rk[None, :]
    sa_ptr = SA + ram * stride_sam
    sbg_ptr = SB + pid_n * stride_sbn
    sbu_ptr = SB + (NTILE + pid_n) * stride_sbn

    accg = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    accu = tl.zeros((BLOCK_M, 128), dtype=tl.float32)

    for kb in tl.range(0, KBLK):
        a = tl.load(a_ptr)
        bg = tl.load(bg_ptr)
        bu = tl.load(bu_ptr)
        sa = tl.load(sa_ptr + kb)
        sbg = tl.load(sbg_ptr + kb)
        sbu = tl.load(sbu_ptr + kb)
        dg = tl.dot(a, tl.trans(bg))
        du = tl.dot(a, tl.trans(bu))
        accg += dg * (sa * sbg)[:, None]
        accu += du * (sa * sbu)[:, None]
        a_ptr += 128
        bg_ptr += 128
        bu_ptr += 128

    gate = accg.to(tl.bfloat16).to(tl.float32)
    up = accu.to(tl.bfloat16).to(tl.float32)
    sil = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    prod = (sil * up).to(tl.bfloat16).to(tl.float32)

    amax = tl.max(tl.abs(prod), axis=1)
    s = tl.maximum(amax * E4M3_RCP, SMIN)
    q = tl.fdiv(prod, s[:, None], ieee_rounding=True)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)

    tl.store(GQ + rm[:, None] * stride_gm + (pid_n * 128 + rn)[None, :],
             q.to(FP8, fp_downcast_rounding="rtne"), mask=rmask[:, None])
    tl.store(GS + rm * stride_gsm + pid_n, s, mask=rmask)


# --------------------------------------------------------------------------
# GEMM 2 (down projection) fused with the routing weight
# --------------------------------------------------------------------------
@triton.jit
def _gemm2_kernel(
    A, SA, B, SB, RW, O,
    M,
    stride_am, stride_sam, stride_bn, stride_sbn, stride_om,
    KBLK: tl.constexpr,
    NTILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_MT: tl.constexpr,
):
    pid = tl.program_id(0)
    width = GROUP_M * NTILE
    gid = pid // width
    off = pid % width
    first = gid * GROUP_M
    gsz = min(NUM_MT - first, GROUP_M)
    pid_m = first + (off % gsz)
    pid_n = off // gsz

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    ram = tl.where(rmask, rm, 0)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)

    a_ptr = A + ram[:, None] * stride_am + rk[None, :]
    b_ptr = B + (pid_n * 128 + rn)[:, None] * stride_bn + rk[None, :]
    sa_ptr = SA + ram * stride_sam
    sb_ptr = SB + pid_n * stride_sbn

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, KBLK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        sa = tl.load(sa_ptr + kb)
        sb = tl.load(sb_ptr + kb)
        acc += tl.dot(a, tl.trans(b)) * (sa * sb)[:, None]
        a_ptr += 128
        b_ptr += 128

    y = acc.to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW + ram, mask=rmask, other=0.0).to(tl.float32)
    out = (y * rw[:, None]).to(tl.bfloat16)
    tl.store(O + rm[:, None] * stride_om + (pid_n * 128 + rn)[None, :],
             out, mask=rmask[:, None])


# --------------------------------------------------------------------------


def _pick_bm(m, ntile, target=512):
    """Choose the M tile so the grid keeps the 256 CUs busy."""
    for bm in (128, 64, 32):
        if triton.cdiv(m, bm) * ntile >= target:
            return bm
    return 32


@torch.no_grad()
def run(hidden_states, routing_weight, gate_up_weight, down_weight):
    M, H = hidden_states.shape
    NGU = gate_up_weight.shape[0]          # 2 * intermediate
    I = NGU // 2                           # intermediate
    dev = hidden_states.device

    hk = H // 128
    ik = I // 128
    nt_gu = I // 128
    nt_dn = H // 128

    hq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    hs = torch.empty((M, hk), dtype=torch.float32, device=dev)
    guq = torch.empty((NGU, H), dtype=torch.float8_e4m3fn, device=dev)
    gus = torch.empty((NGU // 128, hk), dtype=torch.float32, device=dev)
    dnq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
    dns = torch.empty((H // 128, ik), dtype=torch.float32, device=dev)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, ik), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    # ---- quantize activations (1x128) --------------------------------
    BM_Q = 32
    _quant_act_kernel[(triton.cdiv(M, BM_Q), hk)](
        hidden_states, hq, hs,
        M, hidden_states.stride(0), hq.stride(0), hs.stride(0),
        BLOCK_M=BM_Q, num_warps=4, num_stages=2,
    )

    # ---- quantize both weights (128x128) -----------------------------
    nt0 = (NGU // 128) * hk
    nt1 = (H // 128) * ik
    _quant_w2_kernel[(nt0 + nt1,)](
        gate_up_weight, guq, gus,
        gate_up_weight.stride(0), guq.stride(0), gus.stride(0),
        down_weight, dnq, dns,
        down_weight.stride(0), dnq.stride(0), dns.stride(0),
        NK0=hk, NTILES0=nt0, NK1=ik,
        num_warps=8, num_stages=2,
    )

    # ---- gemm 1 + silu + mul + quantize ------------------------------
    bm1 = _pick_bm(M, nt_gu)
    nmt1 = triton.cdiv(M, bm1)
    _gemm1_kernel[(nmt1 * nt_gu,)](
        hq, hs, guq, gus, gq, gs,
        M,
        hq.stride(0), hs.stride(0), guq.stride(0), gus.stride(0),
        gq.stride(0), gs.stride(0),
        KBLK=hk, NTILE=nt_gu, IHALF=I,
        BLOCK_M=bm1, GROUP_M=8, NUM_MT=nmt1,
        num_warps=8, num_stages=2,
    )

    # ---- gemm 2 + routing weight -------------------------------------
    bm2 = _pick_bm(M, nt_dn)
    nmt2 = triton.cdiv(M, bm2)
    _gemm2_kernel[(nmt2 * nt_dn,)](
        gq, gs, dnq, dns, routing_weight, out,
        M,
        gq.stride(0), gs.stride(0), dnq.stride(0), dns.stride(0), out.stride(0),
        KBLK=ik, NTILE=nt_dn,
        BLOCK_M=bm2, GROUP_M=8, NUM_MT=nmt2,
        num_warps=8, num_stages=2,
    )

    return out
