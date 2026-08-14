import os

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused gate + up grouped GEMM  ->  SwiGLU intermediate
#
# For every block of (block-aligned, expert-sorted) rows we load the A tile
# once and use it for BOTH the gate and the up projection, which halves the
# activation traffic relative to two separate GEMMs.
#
# Numerics mirror the reference exactly:
#   g16 = fp16(f32_acc_gate); u16 = fp16(f32_acc_up)
#   s16 = fp16(silu_f32(g16));  inter = fp16(f32(s16) * f32(u16))
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_kernel(
    hidden_ptr,        # [M, H]        fp16
    gate_ptr,          # [E, I, H]     fp16
    up_ptr,            # [E, I, H]     fp16
    inter_ptr,         # [PAD, I]      fp16   (expert-sorted row order)
    sorted_ids_ptr,    # [PAD]         int32  (flat repeated-row id, -1 = pad)
    expert_ids_ptr,    # [NBLK]        int32
    H,
    I,
    stride_he,         # E stride of gate/up  (= I * H)
    NUM_N_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    # n is the fast-varying axis: all n-blocks of one row-block run together
    # so they share the A tile through L2.
    pid_m = pid // NUM_N_BLOCKS
    pid_n = pid % NUM_N_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m)
    valid = tok >= 0
    # Whole block is padding -> nothing to do (skips the weight loads).
    if tl.sum(valid.to(tl.int32)) == 0:
        return

    rows = tl.where(valid, tok, 0) // TOPK

    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = hidden_ptr + rows[:, None].to(tl.int64) * H + offs_k[None, :]
    wbase = e * stride_he + offs_n[:, None].to(tl.int64) * H + offs_k[None, :]
    g_ptrs = gate_ptr + wbase
    u_ptrs = up_ptr + wbase

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(H, BLOCK_K)):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        # Weights are streamed once and never reused -> mark them evict-first
        # so they do not push the reused A tiles out of L2.
        bg = tl.load(g_ptrs, eviction_policy=EVICT)
        bu = tl.load(u_ptrs, eviction_policy=EVICT)
        acc_g = tl.dot(a, tl.trans(bg), acc_g)
        acc_u = tl.dot(a, tl.trans(bu), acc_u)
        a_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K

    # --- reference rounding order -------------------------------------
    g16 = acc_g.to(tl.float16)
    u16 = acc_u.to(tl.float16)
    gf = g16.to(tl.float32)
    s16 = (gf * tl.sigmoid(gf)).to(tl.float16)          # F.silu on a half tensor
    out = (s16.to(tl.float32) * u16.to(tl.float32)).to(tl.float16)

    o_ptrs = inter_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :]
    tl.store(o_ptrs, out, mask=valid[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: grouped down GEMM, routing weight applied in fp16, scattered back
# to the un-sorted [M*TOPK, H] layout.
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    inter_ptr,         # [PAD, I]      fp16 (expert-sorted)
    down_ptr,          # [E, H, I]     fp16
    y_ptr,             # [M*TOPK, H]   fp16
    weight_ptr,        # [M*TOPK]      fp16
    sorted_ids_ptr,    # [PAD]         int32
    expert_ids_ptr,    # [NBLK]        int32
    H,
    I,
    stride_de,         # E stride of down (= H * I)
    NUM_N_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVICT: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N_BLOCKS
    pid_n = pid % NUM_N_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m)
    valid = tok >= 0
    if tl.sum(valid.to(tl.int32)) == 0:
        return

    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = inter_ptr + offs_m[:, None].to(tl.int64) * I + offs_k[None, :]
    b_ptrs = down_ptr + e * stride_de + offs_n[:, None].to(tl.int64) * I + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(I, BLOCK_K)):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        b = tl.load(b_ptrs, eviction_policy=EVICT)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # y (fp16) * topk_weight (fp16) -> fp16, exactly as the reference does
    y16 = acc.to(tl.float16)
    w = tl.load(weight_ptr + tl.where(valid, tok, 0)).to(tl.float32)
    prod = (y16.to(tl.float32) * w[:, None]).to(tl.float16)

    rows = tl.where(valid, tok, 0).to(tl.int64)
    o_ptrs = y_ptr + rows[:, None] * H + offs_n[None, :]
    tl.store(o_ptrs, prod, mask=valid[:, None])


# ---------------------------------------------------------------------------
# Row alignment (sync-free): sort the flat expert ids, then place each
# expert's rows at a BLOCK_M-aligned offset, padding with -1.
# ---------------------------------------------------------------------------
def _align(flat_idx, num_experts, block_m, n_blocks_alloc, device):
    T = flat_idx.numel()
    sv, si = torch.sort(flat_idx)

    ar = torch.arange(num_experts + 1, device=device, dtype=flat_idx.dtype)
    bounds = torch.searchsorted(sv, ar)
    cum_real = bounds[:num_experts]
    counts = bounds[1:] - cum_real

    padded = ((counts + (block_m - 1)) // block_m) * block_m
    cum_pad_incl = torch.cumsum(padded, 0)
    cum_pad = cum_pad_incl - padded

    dest = cum_pad[sv] + (torch.arange(T, device=device, dtype=torch.int64) - cum_real[sv])

    sorted_ids = torch.full(
        (n_blocks_alloc * block_m,), -1, dtype=torch.int32, device=device
    )
    sorted_ids[dest] = si.to(torch.int32)

    block_start = torch.arange(
        n_blocks_alloc, device=device, dtype=torch.int64
    ) * block_m
    expert_ids = torch.searchsorted(cum_pad_incl, block_start, right=True)
    expert_ids = expert_ids.clamp_(max=num_experts - 1).to(torch.int32)

    return sorted_ids, expert_ids


def _pick_block_m(avg_per_expert):
    # Measured on MI355X: BM=64 wins across the whole workload range. Larger
    # row-blocks cut weight re-reads but cost more than they save here,
    # because occupancy and A-tile reuse dominate.
    return 64


_CFG = {
    "BN1": int(os.environ.get("MOE_BN1", 128)),
    "BK1": int(os.environ.get("MOE_BK1", 64)),
    "W1": int(os.environ.get("MOE_W1", 4)),
    "S1": int(os.environ.get("MOE_S1", 1)),
    "BN2": int(os.environ.get("MOE_BN2", 128)),
    "BK2": int(os.environ.get("MOE_BK2", 64)),
    "W2": int(os.environ.get("MOE_W2", 4)),
    "S2": int(os.environ.get("MOE_S2", 1)),
    "BM": int(os.environ.get("MOE_BM", 0)),
    "EV": os.environ.get("MOE_EV", "evict_first"),
}


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
) -> torch.Tensor:
    M, H = hidden_states.shape
    num_experts, I, _ = expert_gate_projs.shape
    topk = topk_idx.shape[1]
    device = hidden_states.device
    T = M * topk

    BLOCK_M = _CFG["BM"] or _pick_block_m(T / num_experts)
    n_blocks_alloc = triton.cdiv(T, BLOCK_M) + num_experts

    flat_idx = topk_idx.reshape(-1)
    sorted_ids, expert_ids = _align(
        flat_idx, num_experts, BLOCK_M, n_blocks_alloc, device
    )

    inter = torch.empty(
        (n_blocks_alloc * BLOCK_M, I), dtype=torch.float16, device=device
    )

    BN1, BK1 = _CFG["BN1"], _CFG["BK1"]
    _gate_up_kernel[(n_blocks_alloc * triton.cdiv(I, BN1),)](
        hidden_states,
        expert_gate_projs,
        expert_up_projs,
        inter,
        sorted_ids,
        expert_ids,
        H,
        I,
        I * H,
        NUM_N_BLOCKS=triton.cdiv(I, BN1),
        TOPK=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BN1,
        BLOCK_K=BK1,
        EVICT=_CFG["EV"],
        num_warps=_CFG["W1"],
        num_stages=_CFG["S1"],
    )

    y = torch.empty((T, H), dtype=torch.float16, device=device)
    BN2, BK2 = _CFG["BN2"], _CFG["BK2"]
    _down_kernel[(n_blocks_alloc * triton.cdiv(H, BN2),)](
        inter,
        expert_down_projs,
        y,
        topk_weight.reshape(-1),
        sorted_ids,
        expert_ids,
        H,
        I,
        H * I,
        NUM_N_BLOCKS=triton.cdiv(H, BN2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BN2,
        BLOCK_K=BK2,
        EVICT=_CFG["EV"],
        num_warps=_CFG["W2"],
        num_stages=_CFG["S2"],
    )

    # matches the reference's fp32-accumulated sum over the topk axis
    return y.view(M, topk, H).sum(dim=1)
