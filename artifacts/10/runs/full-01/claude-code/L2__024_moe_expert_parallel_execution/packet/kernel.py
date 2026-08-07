"""Fused MoE (SwiGLU) forward for AMD MI355X (gfx950) in Triton.

Semantics follow reference.py:
    for each expert e:
        rows = tokens routed to e
        x    = hidden_states[rows]                (fp32)
        g    = silu(x @ gate_w[e].T)              (fp32)
        u    =      x @ up_w[e].T                 (fp32)
        y    = (g*u) @ down_w[e].T                (fp32)
        out[rows] += y * topk_weight              (fp32 accumulation)
    return out.to(bfloat16)

Implementation: expert-sorted token blocking (vLLM-style "align block size"),
one kernel fusing gate+up (sharing the A tile) and one kernel for down-proj
with fused routing-weight scale and fp32 atomic scatter-add.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# gate + up projections fused (shared A tile) + SiLU * up
# --------------------------------------------------------------------------
@triton.jit
def _gate_up_kernel(
    A,
    GW,
    UW,
    C,
    sorted_ids,
    expert_ids,
    num_valid_blocks,
    N,
    K,
    num_valid,
    num_m_blocks,
    stride_am,
    stride_we,
    stride_wn,
    stride_cm,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # L2/MALL-friendly swizzle: walk GROUP_M consecutive (expert-sorted) M blocks
    # against all N tiles before moving on, so one expert's weight panel is reused
    # out of cache instead of re-streamed from HBM.
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = min(num_m_blocks - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    if pid_m >= tl.load(num_valid_blocks):
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    aid = tl.load(sorted_ids + offs_m)
    mask_m = aid < num_valid
    tok = (aid // top_k).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + tok[:, None] * stride_am + offs_k[None, :]

    e = tl.load(expert_ids + pid_m).to(tl.int64)
    w_off = e * stride_we + offs_k[:, None] + offs_n[None, :] * stride_wn
    gw_ptrs = GW + w_off
    uw_ptrs = UW + w_off

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        g = tl.load(gw_ptrs)
        u = tl.load(uw_ptrs)
        acc_g = tl.dot(a, g, acc_g)
        acc_u = tl.dot(a, u, acc_u)
        a_ptrs += BLOCK_K
        gw_ptrs += BLOCK_K
        uw_ptrs += BLOCK_K

    acc_g = acc_g * tl.sigmoid(acc_g)
    out = (acc_g * acc_u).to(C.dtype.element_ty)

    c_ptrs = C + aid[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, out, mask=mask_m[:, None])


# --------------------------------------------------------------------------
# down projection + routing-weight scale + scatter-add
# --------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    A,
    DW,
    OUT,
    TW,
    sorted_ids,
    expert_ids,
    num_valid_blocks,
    N,
    K,
    num_valid,
    num_m_blocks,
    stride_am,
    stride_we,
    stride_wn,
    stride_om,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = min(num_m_blocks - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    if pid_m >= tl.load(num_valid_blocks):
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    aid = tl.load(sorted_ids + offs_m)
    mask_m = aid < num_valid
    aid64 = aid.to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + aid64[:, None] * stride_am + offs_k[None, :]

    e = tl.load(expert_ids + pid_m).to(tl.int64)
    w_ptrs = DW + e * stride_we + offs_k[:, None] + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    scale = tl.load(TW + aid64, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc * scale[:, None]

    tok = (aid64 // top_k)
    o_ptrs = OUT + tok[:, None] * stride_om + offs_n[None, :]
    tl.atomic_add(o_ptrs, acc, mask=mask_m[:, None], sem="relaxed")


# --------------------------------------------------------------------------
# routing / block alignment (no host syncs)
# --------------------------------------------------------------------------
def _align(topk_indices, E, BM):
    dev = topk_indices.device
    flat = topk_indices.reshape(-1)
    n_assign = flat.numel()

    vals, order = torch.sort(flat)
    bounds = torch.searchsorted(
        vals, torch.arange(E + 1, device=dev, dtype=vals.dtype)
    )
    cum_excl = bounds[:E]
    counts = bounds[1:] - cum_excl

    nblocks = (counts + (BM - 1)) // BM
    blk_cum = torch.cumsum(nblocks, 0)
    blk_start = blk_cum - nblocks
    num_valid_blocks = blk_cum[E - 1 : E].to(torch.int32)

    max_blocks = E + (n_assign + BM - 1) // BM

    delta = torch.zeros(max_blocks + 1, dtype=torch.int32, device=dev)
    delta.index_add_(
        0,
        blk_start.clamp_(max=max_blocks),
        torch.ones(E, dtype=torch.int32, device=dev),
    )
    expert_ids = (torch.cumsum(delta[:max_blocks], 0) - 1).to(torch.int32)

    pos = blk_start[vals] * BM + (
        torch.arange(n_assign, device=dev, dtype=torch.int64) - cum_excl[vals]
    )
    sp = torch.full((max_blocks * BM,), n_assign, dtype=torch.int32, device=dev)
    sp[pos] = order.to(torch.int32)

    return sp, expert_ids, num_valid_blocks, max_blocks, n_assign


# (BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M) -- measured on MI355X.
_GU_CFG = (128, 64, 4, 1, 8)
_DN_CFG = (128, 32, 4, 4, 8)


def _pick_bm(n_assign, E):
    return 128


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    T, H = hidden_states.shape
    E, I, _ = gate_proj_weights.shape
    k = topk_indices.shape[1]
    n_assign = T * k

    BM = _pick_bm(n_assign, E)

    sp, expert_ids, nvb, max_blocks, n_assign = _align(topk_indices, E, BM)

    inter = torch.empty((n_assign, I), dtype=torch.bfloat16, device=hidden_states.device)

    BN1, BK1, W1, S1, G1 = _GU_CFG
    _gate_up_kernel[(max_blocks * triton.cdiv(I, BN1),)](
        hidden_states,
        gate_proj_weights,
        up_proj_weights,
        inter,
        sp,
        expert_ids,
        nvb,
        I,
        H,
        n_assign,
        max_blocks,
        hidden_states.stride(0),
        gate_proj_weights.stride(0),
        gate_proj_weights.stride(1),
        inter.stride(0),
        top_k=k,
        BLOCK_M=BM,
        BLOCK_N=BN1,
        BLOCK_K=BK1,
        GROUP_M=G1,
        num_warps=W1,
        num_stages=S1,
    )

    out = torch.zeros((T, H), dtype=torch.float32, device=hidden_states.device)

    BN2, BK2, W2, S2, G2 = _DN_CFG
    _down_kernel[(max_blocks * triton.cdiv(H, BN2),)](
        inter,
        down_proj_weights,
        out,
        topk_weights.reshape(-1),
        sp,
        expert_ids,
        nvb,
        H,
        I,
        n_assign,
        max_blocks,
        inter.stride(0),
        down_proj_weights.stride(0),
        down_proj_weights.stride(1),
        out.stride(0),
        top_k=k,
        BLOCK_M=BM,
        BLOCK_N=BN2,
        BLOCK_K=BK2,
        GROUP_M=G2,
        num_warps=W2,
        num_stages=S2,
    )

    return out.to(torch.bfloat16)
